import abc
import tensorflow as tf
from copy import deepcopy
from typing import Callable, List, Optional

from maxent.base import MaxEntModel, Particles, get_grads_and_vars, Callback
from maxent.utils import inplace, infinity_norm, quantize_tensor, random


class State(Particles):
  """Seperates each particle into ambient part and latent part."""

  def __init__(self, ambient: tf.Tensor, latent: tf.Tensor):
    self.ambient = ambient
    self.latent = latent


class Distribution(abc.ABC):

  @abc.abstractmethod
  def sample(self, seed: int) -> tf.Tensor:
    return NotImplemented

  @abc.abstractproperty
  def prob_argmax(self) -> tf.Tensor:
    return NotImplemented


class BoltzmannMachine(MaxEntModel):

  @abc.abstractmethod
  def gibbs_sampling(self, state: State) -> State:
    """For doing contrastive divergence."""
    return NotImplemented

  @abc.abstractmethod
  def activate(self, state: State) -> State:
    return NotImplemented

  @abc.abstractmethod
  def get_latent_given_ambient(self, ambient: tf.Tensor) -> Distribution:
    return NotImplemented


class Initializer(abc.ABC):
  """Initializer for `BoltzmannMachine`."""

  @abc.abstractproperty
  def ambient_ambient_kernel(self):
    return NotImplemented

  @abc.abstractproperty
  def ambient_bias(self):
    return NotImplemented

  @abc.abstractproperty
  def latent_latent_kernel(self):
    return NotImplemented

  @abc.abstractproperty
  def latent_bias(self):
    return NotImplemented

  @abc.abstractproperty
  def ambient_latent_kernel(self):
    return NotImplemented


@inplace('bm, e.t.c.')
def train(bm: BoltzmannMachine,
          optimizer: tf.optimizers.Optimizer,
          dataset: tf.data.Dataset,
          fantasy_state: State,
          mc_steps: int = 1,
          callbacks: List[Callback] = None):
  """Returns the final fantasy state."""
  for step, real_ambient in enumerate(dataset):
    grads_and_vars = _get_grads_and_vars(
        bm, real_ambient, fantasy_state, bm.seed)
    optimizer.apply_gradients(grads_and_vars)
    fantasy_state = contrastive_divergence(bm, fantasy_state, mc_steps)

    if callbacks:
      real_latent = bm.get_latent_given_ambient(real_ambient).sample(bm.seed)
      real_state = State(real_ambient, real_latent)
      for callback in callbacks:
        callback(step, real_state, fantasy_state, grads_and_vars)

  return fantasy_state


def _get_grads_and_vars(bm: BoltzmannMachine,
                        real_ambient: tf.Tensor,
                        fantasy_state: State,
                        seed: int):
  """For applying `tf.optimizers.Optimizer.apply_gradients` method."""
  real_latent = bm.get_latent_given_ambient(real_ambient).sample(seed)
  real_state = State(real_ambient, real_latent)
  return get_grads_and_vars(bm, real_state, fantasy_state)


def contrastive_divergence(bm: BoltzmannMachine,
                           fantasy_state: State,
                           mc_steps: int):
  """Returns the final fantasy state."""
  for _ in tf.range(mc_steps):
    fantasy_state = bm.gibbs_sampling(fantasy_state)
  return fantasy_state


def _relax(activate: Callable[[State], State],
           state: State,
           max_step: int,
           tolerance: float):
  """Evolves the dynamics until the two adjacent states are the same, and
  returns the final ambient and the final step of evolution.

  The miximum step of the evolution is `max_step`, until then stop the
  evolution, regardless whether it has been relaxed or not.

  The word "same" means that the L-infinity norm of the difference is smaller
  than the `tolerance`.
  """

  def shall_stop(state: State, new_state: State):
    if infinity_norm(new_state.ambient - state.ambient) > tolerance:
      return False
    if infinity_norm(new_state.latent - state.latent) > tolerance:
      return False
    return True

  step = 1
  while tf.less_equal(step, max_step):
    new_state = activate(state)
    if shall_stop(state, new_state):
      break
    state = new_state
    step += 1
  return state, step


def relax(bm: BoltzmannMachine,
          state: State,
          max_step: int,
          tolerance: float):
  return _relax(bm.activate, state, max_step, tolerance)


def relax_ambient(bm: BoltzmannMachine,
                  ambient: tf.Tensor,
                  max_step: int,
                  tolerance: float):
  latent = bm.get_latent_given_ambient(ambient).prob_argmax
  state = State(ambient, latent)
  return relax(bm, state, max_step, tolerance)


def get_reconstruction_error(bm: BoltzmannMachine,
                             ambient: tf.Tensor,
                             norm: Callable[[tf.Tensor], float]):
  """Analygy to the computation of reconstruction error of the restricted."""
  latent = bm.get_latent_given_ambient(ambient).prob_argmax
  recon_ambient = bm.activate(State(ambient, latent)).ambient
  return norm(recon_ambient - ambient)


def quantize(bm: BoltzmannMachine, precision: float, return_int: bool):
  quantized_bm = deepcopy(bm)
  for i, (param, _) in enumerate(bm.params_and_obs):
    quantized_bm.params_and_obs[i][0].assign(
        quantize_tensor(param, precision, return_int))
  return quantized_bm


class UpdateWithMasks:
  """For type-hinting.

  Updates the state `state` with the ambient mask `ambient_mask` and the
  latent mask `latent mask`.

  When the `ambient_mask` is `None`, then update without ambient mask. The
  same for the `latent_mask`.
  """

  def __call__(self,
               state: State,
               ambient_mask: Optional[tf.Tensor],
               latent_mask: Optional[tf.Tensor]):
    return NotImplemented


def async_update(update_with_masks: UpdateWithMasks,
                 state: State,
                 sync_ratio: float,
                 seed: int):
  """Simulates the async element-wise update by partially sync update.

  Precisely, update `sync_ratio` percent elements for both ambient and latent
  at each iteration. And end up iterating when all elements have been updated.
  Only update once for each element.

  The argument `sync_ratio` is in range (0, 1].
  """
  if sync_ratio == 1:
    return update_with_masks(state, None, None)

  def get_mask(pre_mask: tf.Tensor, iter_step: int) -> tf.Tensor:
    mask_ratio = 1 - (1 - sync_ratio) ** iter_step
    mask = tf.where(random(pre_mask.shape, seed) < mask_ratio, 1., 0.)
    mask = tf.where(pre_mask > 0, pre_mask, mask)
    return mask

  ambient_mask = tf.zeros_like(state.ambient)
  latent_mask = tf.zeros_like(state.latent)
  iter_step = 1
  while True:
    ambient_mask = get_mask(ambient_mask, iter_step)
    latent_mask = get_mask(latent_mask, iter_step)
    state = update_with_masks(state, ambient_mask, latent_mask)
    iter_step += 1
    if tf.reduce_min(ambient_mask) * tf.reduce_min(latent_mask) > 0:
      # all element have been updated.
      break
  return state
