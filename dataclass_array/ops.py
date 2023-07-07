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

"""Operation utils."""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, Optional  # pylint: disable=g-multiple-import

from dataclass_array import array_dataclass
from dataclass_array.typing import Array, DcT  # pylint: disable=g-multiple-import
from dataclass_array.utils import np_utils
from etils import enp
from etils import epy


def _ops_base(
    arrays: Iterable[DcT],
    *,
    axis: int,
    array_fn: Callable[
        [
            enp.NpModule,
            int,
            Any,  # array_dataclass._ArrayField[Array['*din']],
        ],
        Array['*dout'],
    ],
    dc_fn: Optional[
        Callable[
            [
                enp.NpModule,
                int,
                Any,  # array_dataclass._ArrayField[DcT],
            ],
            DcT,
        ]
    ],
) -> DcT:
  """Base function for all ops."""
  arrays = list(arrays)
  first_arr = arrays[0]

  if not isinstance(first_arr, array_dataclass.DataclassArray):
    raise TypeError(
        '`dca.stack` expect list of `dca.DataclassArray`. Got '
        f'{type(first_arr)}'
    )

  # This might have some edge cases if user try to stack subclasses
  types = epy.groupby(
      arrays,
      key=type,
      value=lambda x: type(x).__name__,
  )
  if False in types:
    raise TypeError(
        f'v3.stack got conflicting types as input: {list(types.values())}'
    )

  xnp = first_arr.xnp
  # If axis < 0, normalize the axis such as the last axis is before the inner
  # shape
  axis = np_utils.to_absolute_axis(axis, ndim=first_arr.ndim + 1)

  # Iterating over only the fields of the `first_arr` will skip optional fields
  # if those are not set in `first_arr`, even if they are present in others.
  # But is consistent with `jax.tree_map`:
  # jax.tree_map(lambda x, y: x+y, (None, 10), (1, 2)) == (None, 12)
  # Similarly, static values will be the ones from the first element.
  merged_arr = first_arr._map_field(  # pylint: disable=protected-access
      array_fn=functools.partial(array_fn, xnp, axis),
      dc_fn=functools.partial(dc_fn, xnp, axis),
  )
  return merged_arr


def stack(
    arrays: Iterable[DcT],  # list[_DcT['*shape']]
    *,
    axis: int = 0,
) -> DcT:  # _DcT['len(arrays) *shape']:
  """Stack dataclasses together."""
  return _ops_base(
      arrays,
      axis=axis,
      array_fn=lambda xnp, axis, f: xnp.stack(  # pylint: disable=g-long-lambda
          [getattr(arr, f.name) for arr in arrays], axis=axis
      ),
      dc_fn=lambda xnp, axis, f: stack(  # pylint: disable=g-long-lambda
          [getattr(arr, f.name) for arr in arrays],
          axis=axis,
      ),
  )


def concat(arrays: Iterable[DcT], *, axis: int = 0) -> DcT:
  """Concatenate dataclasses together."""
  return _ops_base(
      arrays,
      axis=axis,
      array_fn=lambda xnp, axis, f: xnp.concatenate(  # pylint: disable=g-long-lambda
          [getattr(arr, f.name) for arr in arrays], axis=axis
      ),
      dc_fn=lambda xnp, axis, f: concat(  # pylint: disable=g-long-lambda
          [getattr(arr, f.name) for arr in arrays],
          axis=axis,
      ),
  )
