"""Collections of util functions and classes."""

import abc
import numpy as np
import tensorflow as tf
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set
from functools import wraps


def annotate(description: str):
  """The decorated function is the same as the original, except for appending
  the string `description` into the docstring."""

  def decorator(func):

    @wraps(func)
    def decorated(*args, **kwargs):
      return func(*args, **kwargs)

    origin_doc = decorated.__doc__
    if origin_doc is None:
      new_doc = description
    else:
      new_doc = '\n\n'.join([origin_doc, description])
    if not new_doc.endswith('\n'):
      new_doc += '\n'
    decorated.__doc__ = new_doc

    return decorated

  return decorator


def inplace(inplace_args: str):
  """Returns a decorator hinting in-place operation on the arguments of
  the decorated function.

  The argument `inplace_args` represents a description of the in-place
  arguments.
  """
  description = (
      'CAUTION:\n\t'
      f'In-place function on the argument(s) {inplace_args}!'
  )
  return annotate(description)


def inner(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Inner product along the last axis."""
  return tf.reduce_sum(x * y, axis=-1)


def outer(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Outer product along the last axis of `x` and `y`. The output tensor has
  shape [... x.shape[-1], y.shape[-1]].
  """
  return tf.expand_dims(x, axis=-1) * tf.expand_dims(y, axis=-2)


def random(shape: List[int], seed: int) -> tf.Tensor:
  return tf.random.uniform(shape=shape, minval=0., maxval=1., seed=seed)


def expect(x: tf.Tensor) -> tf.Tensor:
  """Expectation along the first (i.e. batch) axis."""
  return tf.reduce_mean(x, axis=0)


def infinity_norm(x: tf.Tensor):
  norm: tf.Tensor = tf.reduce_max(tf.abs(x))
  return norm


def create_variable(name: str,
                    shape: List[int],
                    initializer: tf.initializers.Initializer,
                    dtype: str = 'float32',
                    **variable_kwargs):
  """
  Parameters
  ----------
  **variable_kwargs
    kwargs of `tf.Variable`, excluding `name`, `shape`, `init_value`,
    and `dtype`.
  """
  init_value = initializer(shape, dtype)
  return tf.Variable(init_value, name=name, **variable_kwargs)


class History:
  """Util for logging the internal information in a training process."""

  def __init__(self):
    self.logs = defaultdict(dict)

  def log(self, step: int, key: str, value: object):
    try:  # maybe a tf.Tensor
      value = value.numpy()
    except AttributeError:
      pass
    self.logs[step][key] = value

  def show(self, step: int, keys: List[str] = None):
    """Returns the string to show."""
    if keys is None:
      keys = list(self.logs[step])

    aspects = []
    for k in keys:
      v = self.logs[step].get(k, None)
      if isinstance(v, (float, np.floating)):
        v = f'{v:.5f}'
      elif isinstance(v, str):
        pass
      else:
        raise ValueError(f'Type {type(v)} is temporally not supported.')
      aspects.append(f'{k}: {v}')

    show_str = ' - '.join([f'step: {step}'] + aspects)
    return show_str


class ComposedConstraint(tf.keras.constraints.Constraint):

  def __init__(self, constraints: List[tf.keras.constraints.Constraint]):
    self.constraints = constraints
  
  def __call__(self, kernel: tf.Tensor):
    result = kernel
    for constraint in self.constraints:
      result = constraint(result)
    return result


class Connections:
  pass


class NoConnection(Connections):
  pass


class DenseConnections(Connections):
  pass


class SparseConnections(Connections):

  def __init__(self):
    self._connected: Dict[int, Set[int]] = defaultdict(set)
  
  def __iter__(self):
    for i, js in self._connected.items():
      for j in js:
        yield i, j
  
  def connect(self, i: int, j: int):
    self._connected[i].add(j)


class SparsityConstraint(tf.keras.constraints.Constraint):

  def __init__(self, connections: SparseConnections):
    self.connections = connections

    self.mask = None
    self.built = False

  def __call__(self, kernel: tf.Tensor):
    if not self.built:
      self.build(kernel.shape, kernel.dtype)
    return self.mask * kernel

  def build(self, shape, dtype):
    mask = np.zeros(shape)
    for i, j in self.connections:
      mask[i, j] = 1
    self.mask = tf.convert_to_tensor(mask, dtype=dtype)
    self.built = True


def get_random_connections(size_a: int,
                           size_b: int,
                           sparsity: float):
  """If sparsity is unit, then connection all. And if vanishing, then no
  connection."""
  connections = SparseConnections()
  for i in range(size_a):
    for j in range(size_b):
      if np.random.random() < sparsity:
        connections.connect(i, j)
  return connections


class SymmetricDiagonalVanishingConstraint(tf.keras.constraints.Constraint):

  def __call__(self, kernel: tf.Tensor):
    num_rows, num_columns = kernel.shape
    assert num_rows == num_columns
    kernel = 0.5 * (kernel + tf.transpose(kernel))
    kernel = tf.linalg.set_diag(kernel, tf.zeros([num_rows]))
    return kernel


class MovingAverage(abc.ABC):

  @abc.abstractmethod
  def __call__(self, x: tf.Tensor, axis: int) -> tf.Tensor:
    return NotImplemented


class ExponentialMovingAverage(MovingAverage):

  def __init__(self, weight: float):
    self.weight = weight

  def __call__(self, x: tf.Tensor, axis: int):
    x = tf.unstack(x, axis=axis)
    s = x[0]
    smoothed = []
    for xi in x:
      s = s * self.weight + (1 - self.weight) * xi
      smoothed.append(s)
    return tf.stack(smoothed, axis=axis)


def quantize_tensor(x: tf.Tensor, precision: float, return_int: bool):
  """If `return_int`, then returns float type integer."""
  quantized = tf.cast(tf.cast(x / precision, 'int32'), x.dtype)
  return quantized if return_int else quantized * precision


# TODO: add docstring.
def get_supremum(condition: Callable[[float], bool],
                 init: float,
                 minval: float,
                 maxval: float,
                 eps: float):
  assert condition(init)
  x = init
  while maxval - minval > eps:
    if condition(x):
      minval = x
    else:
      maxval = x
    x = minval + 0.5 * (maxval - minval)
  return minval if condition(minval) else None


def update_with_mask(new: tf.Tensor,
                     old: tf.Tensor,
                     mask: Optional[tf.Tensor]):
  """While element-wisely updating from the old tensor `old` to the new `new`,
  only the positions on which `mask > 0` will be updated.

  When the `mask` is `None`, then returns the new `new` directly.
  """
  result = new
  if mask is not None:
    result: tf.Tensor = tf.where(mask > 0, result, old)
  return result
