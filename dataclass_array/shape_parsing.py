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

"""Shape and type parsing util."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional, Tuple, Union

from dataclass_array.utils import file_utils
import lark

# TODO(epot): Once stable, some of this should be moved in `etils.array_types`

_Dim = Optional[int]
_Shape = Tuple[_Dim, ...]

_DimAst = Union[int, None, '_NamedDim', '_VarDim']
_ShapeAst = Tuple[_DimAst, ...]


@dataclasses.dataclass(frozen=True, eq=True)
class _NamedDim:
  name: str


@dataclasses.dataclass(frozen=True, eq=True)
class _VarDim:
  name: str


@dataclasses.dataclass
class _Constant:
  value: Any

  def __call__(self, _) -> Any:
    return self.value


class _TreeShapeTransformer(lark.Transformer):
  """Transform the tree into value."""

  shape = tuple

  named_dim = lark.v_args(inline=True)(_NamedDim)
  var_dim = lark.v_args(inline=True)(_VarDim)
  UNKNOWN_DIM = _Constant(None)
  ELLIPSIS_DIM = _Constant(_VarDim(name='_'))
  STATIC_DIM = int

  CNAME = str


class ShapeParser:
  """Shape parser."""

  @classmethod
  @functools.lru_cache()
  def singleton(cls) -> ShapeParser:
    """Factory creating a unique global instance."""
    return cls()

  def __init__(self):
    grammar_path = file_utils.dca_path() / 'shape_grammar.lark'
    self.parser = lark.Lark(grammar_path.read_text())
    self.transformer = _TreeShapeTransformer()

  def parse(self, shape_str: str) -> _ShapeAst:
    return self.transformer.transform(self.parser.parse(shape_str))


def get_inner_shape(shape_str: str) -> _Shape:
  """Parse the string and extract the inner shape."""
  parser = ShapeParser.singleton()
  shape = parser.parse(shape_str)

  # TODO(epot): Reraise typing with `shape` debug message
  if not shape or not isinstance(shape[0], _VarDim):
    raise ValueError(
        f'Shape {shape_str!r} should start by `...` or `*shape` (e.g. '
        "`f32['*shape 3']`)"
    )

  inner_shape = shape[1:]
  # Currently, `_NamedDim` are accepted but consistency isn't checked across
  # fields
  inner_shape = tuple(
      None if isinstance(s, _NamedDim) else s for s in inner_shape
  )
  if not all(isinstance(dim, (int, type(None))) for dim in inner_shape):
    raise ValueError('Only static or None dimensions supported.')

  return inner_shape  # pytype: disable=bad-return-type
