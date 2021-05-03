import abc
import tensorflow as tf
from typing import List, Tuple

from maxent.utils import expect


class Particles(abc.ABC):
  """A batch of particles of the MaxEnt model."""


class Operator(abc.ABC):

  @abc.abstractmethod
  def __call__(self, particles: Particles) -> tf.Tensor:
      return NotImplemented


class MaxEntModel(abc.ABC):

  @abc.abstractproperty
  def params_and_ops(self) -> List[Tuple[tf.Tensor, Operator]]:
    return NotImplemented


def get_grads_and_vars(max_ent_model: MaxEntModel,
                       real_particles: Particles,
                       fantasy_particles: Particles):
  grads_and_vars: List[Tuple[tf.Tensor, tf.Tensor]] = []
  for param, op in max_ent_model.params_and_ops:
    grad_param = expect(op(fantasy_particles)) - expect(op(real_particles))
    grads_and_vars.append((grad_param, param))
  return grads_and_vars
