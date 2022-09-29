# Copyright 2022 The dataclass_array Authors.
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

"""Typing annotation parsing util."""

from __future__ import annotations

import sys
import types
import typing
from typing import Any, Callable, Optional

from dataclass_array import array_dataclass
from dataclass_array import field_utils
from dataclass_array.typing import TypeAlias
from etils import array_types as array_types_lib
import typing_extensions  # TODO(py38): Remove

_LeafFn = Callable[[TypeAlias], None]

_NoneType = type(None)


def _visit_leaf(hint: TypeAlias, leaf_fn: _LeafFn):
  """Leaf node."""
  if hint == _NoneType:  # Normalize `None`
    hint = None
  return leaf_fn(hint)


def _visit_union(hint: TypeAlias, leaf_fn: _LeafFn):
  """Recurse in `Union[x, y]`, `x | y`, or `Optional[x]`."""
  item_hints = typing_extensions.get_args(hint)
  for item_hint in item_hints:
    _visit(item_hint, leaf_fn)


def _visit(hint: TypeAlias, leaf_fn: _LeafFn):
  """Recurse in the type annotation tree."""
  origin = typing_extensions.get_origin(hint)
  visit_fn = _ORIGIN_TO_VISITOR.get(origin, _visit_leaf)
  visit_fn(hint, leaf_fn)


# Currently, only support `Union` and `Optional` but could be extended
# to `dict`, `list`,...
_ORIGIN_TO_VISITOR = {
    typing.Union: _visit_union,
    None: _visit_leaf,  # Default origin
}
if sys.version_info >= (3, 10):
  _ORIGIN_TO_VISITOR[types.UnionType] = _visit_union  # In Python 3.10+: x | y


def _get_leaf_types(hint: TypeAlias) -> list[type[Any]]:
  """Extract the inner list of the types (`Optional[A] -> [A, None]`)."""
  all_types = []

  def _collect_leaf_types(hint):
    all_types.append(hint)

  _visit(hint, leaf_fn=_collect_leaf_types)

  return all_types


def get_array_type(hint: TypeAlias) -> Optional[Any]:
  """Returns the array type, or `None` if no type was detected.

  Example:

  ```python
  get_array_type(f32[..., 3]) -> f32[..., 3]
  get_array_type(dca.Ray) -> dca.Ray['...']
  get_array_type(Optional[dca.Ray]) -> dca.Ray['...']
  get_array_type(dca.Ray | dca.Camera | None) -> dca.DataclassArray
  get_array_type(Any) -> None  # Any not an array type
  get_array_type(dca.Ray | int) -> None  # int not an array type
  get_array_type(list[dca.Ray]) -> None  # list not an array type
  get_array_type(dca.Ray | f32['... 3']) -> NotImplementedError (unsupported)
  ```

  Args:
    hint: The typing annotation

  Returns:
    The array type, or `None` if not type was detected
  """
  leaf_types = _get_leaf_types(hint)
  # Filter `None` element (e.g. `Optional[dca.Ray]`)
  leaf_types = [l for l in leaf_types if l is not None]
  if not leaf_types:
    return None

  dc_types = []
  array_types = []
  other_types = []
  for leaf in leaf_types:
    if field_utils.DataclassWithShape.is_dca(leaf):
      dc_types.append(field_utils.DataclassWithShape.from_hint(leaf))
    elif isinstance(leaf, array_types_lib.ArrayAliasMeta):
      array_types.append(leaf)
    else:
      other_types.append(leaf)

  if other_types:  # Non-array type
    return None
  if array_types and dc_types:
    raise NotImplementedError(
        f'{hint} mix dataclass and array. Please open an issue if you need '
        'this feature.'
    )
  if dc_types:
    if len(dc_types) > 1:
      # Validate the inner shape
      common_shapes = {x.shape for x in dc_types}
      if len(common_shapes) != 1:
        raise NotImplementedError(
            f'{hint} mix dataclass with different inner shape. Please open an '
            'issue if you need this feature.'
        )
      (common_shape,) = common_shapes
      return field_utils.DataclassWithShape(
          cls=array_dataclass.DataclassArray,
          shape=common_shape,
      )
    else:
      return dc_types[0]
  if array_types:
    if len(array_types) > 1:
      raise NotImplementedError(
          f'{hint} mix multiple array types. Please open an issue if you need '
          'this feature.'
      )
    else:
      return array_types[0]
