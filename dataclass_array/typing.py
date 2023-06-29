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

"""Types utils."""

from __future__ import annotations

import typing
from typing import Any, Optional, Tuple, Type, TypeVar, Union

from etils import enp

if typing.TYPE_CHECKING:
  from dataclass_array import array_dataclass  # pylint: disable=g-bad-import-order,unused-import

# ======== Array types (alias of `enp.typing`) ========

ArrayAliasMeta = enp.typing.ArrayAliasMeta
ArrayLike = enp.typing.ArrayLike

Array = enp.typing.Array
FloatArray = enp.typing.FloatArray
IntArray = enp.typing.IntArray
BoolArray = enp.typing.BoolArray
StrArray = enp.typing.StrArray

ui8 = enp.typing.ui8
ui16 = enp.typing.ui16
ui32 = enp.typing.ui32
ui64 = enp.typing.ui64
i8 = enp.typing.i8
i16 = enp.typing.i16
i32 = enp.typing.i32
i64 = enp.typing.i64
f16 = enp.typing.f16
f32 = enp.typing.f32
f64 = enp.typing.f64
complex64 = enp.typing.complex64
complex128 = enp.typing.complex128
bool_ = enp.typing.bool_

# ======== Dataclass array specific types ========

TypeAlias = Any

Shape = Tuple[int, ...]
DynamicShape = Tuple[Optional[int], ...]
# One or multiple axis. `None` indicate all axes. This is the type of
# .mean(axis=...)
Axes = Union[None, Shape, int]

DTypeArg = Type[
    Union[
        int,
        float,
        # TODO(epot): Add `np.typing.DTypeLike` once numpy version is updated
        'array_dataclass.DataclassArray',
    ]
]

DcT = TypeVar('DcT', bound='array_dataclass.DataclassArray')

# Typing representing `xnp.ndarray` or `dca.DataclassArray`
DcOrArray = Union[FloatArray[...], 'array_dataclass.DataclassArray']
DcOrArrayT = TypeVar('DcOrArrayT')
