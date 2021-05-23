import abc
import tensorflow as tf
from copy import deepcopy
from typing import List, Tuple

from maxent.utils import expect, quantize_tensor


class Particles(abc.ABC):
  """A batch of particles of the MaxEnt model."""


class Observable(abc.ABC):

  @abc.abstractmethod
  def __call__(self, particles: Particles) -> tf.Tensor:
      return NotImplemented


class MaxEntModel(abc.ABC):

  @abc.abstractproperty
  def params_and_obs(self) -> List[Tuple[tf.Tensor, Observable]]:
    return NotImplemented


def get_grads_and_vars(max_ent_model: MaxEntModel,
                       real_particles: Particles,
                       fantasy_particles: Particles):
  grads_and_vars: List[Tuple[tf.Tensor, tf.Tensor]] = []
  for param, ob in max_ent_model.params_and_obs:
    grad_param = expect(ob(fantasy_particles)) - expect(ob(real_particles))
    grads_and_vars.append((grad_param, param))
  return grads_and_vars


class Callback(abc.ABC):

  @abc.abstractmethod
  def __call__(self,
               step: int,
               real_particles: Particles,
               fantasy_particles: Particles,
               grads_and_vars: List[Tuple[tf.Tensor, tf.Tensor]],
               ) -> None:
    return NotImplemented
