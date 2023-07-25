# Copyright 2023 The dataclass_array Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numpy utils.

And utils intended to work on both `xnp.ndarray` and `dca.DataclassArray`.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from dataclass_array import array_dataclass
from dataclass_array.typing import Axes, DTypeArg, DcOrArrayT, Shape  # pylint: disable=g-multiple-import,g-importing-member
from etils import enp
from etils.array_types import Array, ArrayLike  # pylint: disable=g-multiple-import,g-importing-member

# Maybe some of those could live in `enp` ?


def size_of(shape: Shape) -> int:
  """Returns the size associated with the shape."""
  # TODO(b/198633198): Warning: In TF `bool(shape) == True` for `shape==()`
  if not len(shape):  # pylint: disable=g-explicit-length-test
    size = 1  # Special case because `np.prod([]) == 1.0`
  else:
    size = enp.lazy.np.prod(shape)
  return size


def get_xnp(x: Any, *, strict: bool = True) -> enp.NpModule:
  """Returns the np module associated with the given array or DataclassArray."""
  if isinstance(x, array_dataclass.DataclassArray):
    xnp = x.xnp
  elif enp.lazy.is_array(x, strict=strict):
    xnp = enp.lazy.get_xnp(x, strict=strict)
  else:
    raise TypeError(
        f'Unexpected array type: {type(x)}. Could not infer numpy module.'
    )
  return xnp


def is_array(
    x: Any,
    *,
    xnp: Optional[enp.NpModule] = None,
) -> bool:
  """Returns whether `x` is an array or DataclassArray.

  Args:
    x: array to check
    xnp: If given, return False if the array is from a different numpy module.

  Returns:
    True if `x` is `xnp.ndarray` or `dca.DataclassArray`
  """
  try:
    infered_xnp = get_xnp(x)
  except TypeError:
    return False
  else:
    if xnp is None:
      return True
    else:
      return infered_xnp is xnp


def asarray(
    x: Union[DcOrArrayT, ArrayLike[Array['...']]],
    *,
    xnp: enp.NpModule = None,
    dtype: Optional[DTypeArg] = None,
    optional: bool = False,
    cast_dtype: bool = True,
) -> DcOrArrayT:
  """Convert `list` to arrays.

  * Validate that x is either `np` or `xnp` (e.g. `np->jnp`, `np->tf` works,
    but not `jnp->tf`, `jnp->np`,...)
  * Dataclass arrays are forwarded.

  Args:
    x: array to check
    xnp: If given, raise an error if the array is from a different numpy module.
      strict
    dtype: If given, cast the array to the dtype
    optional: If True, `x` can be None
    cast_dtype: If False, do not cast the x dtype

  Returns:
    True if `x` is `xnp.ndarray` or `dca.DataclassArray`
  """
  # Potentially forward `None`, if optional is accepted
  if x is None:
    if optional:
      return x
    else:
      raise ValueError('Expected array, got `None`')

  _assert_valid_xnp_cast(from_=get_xnp(x, strict=False), to=xnp)

  # TODO(epot): Could have a DataclassDtype to unify the two cases ?

  # Handle DataclassArray
  if isinstance(x, array_dataclass.DataclassArray):
    # If dtype is given, validate this match
    if dtype is not None and (
        not isinstance(dtype, type) or not isinstance(x, dtype)
    ):
      raise TypeError(f'Expected {dtype}. Got: {type(x)}')
    return x.as_xnp(xnp)

  # Handle ndarray
  dtype = enp.dtypes.DType.from_value(dtype)
  return dtype.asarray(x, xnp=xnp, casting='all' if cast_dtype else 'none')


def _assert_valid_xnp_cast(from_: enp.NpModule, to: enp.NpModule) -> None:
  """Only `np` -> `xnp` conversion is accepted."""
  if from_ is not enp.lazy.np and from_ is not to:
    raise TypeError(f'Expected {to.__name__} got {from_.__name__}')


def to_absolute_axis(axis: Axes, *, ndim: int) -> Axes:
  """Normalize the axis to absolute value.

  Example for self.shape == (x0, x1, x2, x3):

  ```
  to_absolute_axis(None) == (0, 1, 2, 3)
  to_absolute_axis(0) == 0
  to_absolute_axis(-1) == 3
  to_absolute_axis(-2) == 2
  to_absolute_axis((-1, -2)) == (3, 2)
  ```

  Args:
    axis: Axis to normalize
    ndim: Number of dimensions

  Returns:
    The new axis
  """
  if axis is None:
    return tuple(range(ndim))
  elif isinstance(axis, int):
    if axis >= ndim or axis < -ndim:
      raise enp.lazy.np.AxisError(
          axis=axis,
          ndim=ndim,
          # msg_prefix=
          # f'For {self.__class__.__qualname__} with shape={self.shape}',
      )
    elif axis < 0:
      return ndim + axis
    else:
      return axis
  elif isinstance(axis, tuple):
    if not all(isinstance(dim, int) for dim in axis):
      raise ValueError(f'Invalid axis={axis}')
    return tuple(to_absolute_axis(dim, ndim=ndim) for dim in axis)  # pytype: disable=bad-return-type  # always-use-return-annotations
  else:
    raise TypeError(f'Unexpected axis type: {type(axis)} {axis}')


def to_absolute_einops(shape_pattern: str, *, nlastdim: int) -> str:
  """Convert the einops to absolute."""
  # Nested dataclass might already have shape set.
  offset = 0
  while _einops_dim_name(offset) in shape_pattern:
    offset += 1
  last_dims = [_einops_dim_name(i + offset) for i in range(nlastdim)]
  last_dims = ' '.join(last_dims)
  before, after = shape_pattern.split('->')
  before = f'{before} {last_dims} '
  after = f'{after} {last_dims}'
  return '->'.join([before, after])


def _einops_dim_name(i: int) -> str:
  return f'arr__{i}'
