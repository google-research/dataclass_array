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

"""Field utils."""

from __future__ import annotations

import dataclasses

from dataclass_array import array_dataclass
from dataclass_array.typing import TypeAlias
from etils import epy
from etils.array_types import Array
import typing_extensions


@dataclasses.dataclass(eq=True, frozen=True)
class ShapeAnnotation:
  """Annotations for `Ray[''] == Annotated[Ray, ShapeAnnotation('')]`."""

  shape: str

  def __post_init__(self):
    # Normalize shape
    # Might be a cleaner way to do this
    super().__setattr__('shape', Array[self.shape].shape)


@dataclasses.dataclass(eq=True, frozen=True)
class DataclassWithShape:
  """Structure which represent `Ray['h w']`."""

  cls: type[array_dataclass.DataclassArray]
  shape: str

  @classmethod
  def from_hint(cls, hint: TypeAlias) -> DataclassWithShape:
    """Factory to create the `DataclassWithShape` from `MyDca['h w']`."""
    assert cls.is_dca(hint)

    # Extract the shape
    shape = '...'
    if typing_extensions.get_origin(hint) is typing_extensions.Annotated:
      shapes = [a for a in hint.__metadata__ if isinstance(a, ShapeAnnotation)]  # pytype: disable=attribute-error
      if len(shapes) > 1:
        raise ValueError(f'Conflicting annotations for {hint}')
      elif len(shapes) == 1:
        (shape,) = shapes
        shape = shape.shape

      hint = hint.__origin__
    return cls(cls=hint, shape=shape)

  @classmethod
  def is_dca(cls, hint: TypeAlias) -> bool:
    if typing_extensions.get_origin(hint) is typing_extensions.Annotated:
      hint = hint.__origin__
    return epy.issubclass(hint, array_dataclass.DataclassArray)
