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

"""Tests for type_parsing."""

from __future__ import annotations

from typing import Optional, List, Union

import dataclass_array as dca
from dataclass_array import type_parsing
from dataclass_array.typing import f32, FloatArray  # pylint: disable=g-multiple-import
import pytest

_DS = dca.field_utils.DataclassWithShape
Ray = dca.testing.Ray


class Camera(dca.DataclassArray):
  pos: FloatArray[..., 3]
  dir: FloatArray[..., 3]


@pytest.mark.parametrize(
    'hint, expected',
    [
        (int, [int]),
        (Ray, [Ray]),
        (Ray['h w'], [Ray['h w']]),
        (Ray[..., 3], [Ray[..., 3]]),
        (Union[Ray, int], [Ray, int]),
        (Union[Ray['h w'], int], [Ray['h w'], int]),
        (Union[Ray, int, None], [Ray, int, None]),
        (Optional[Ray], [Ray, None]),
        (Optional[Union[Ray, int]], [Ray, int, None]),
        (List[int], [List[int]]),
        (f32[3, 3], [f32[3, 3]]),
    ],
)
def test_get_leaf_types(hint, expected):
  assert type_parsing._get_leaf_types(hint) == expected


@pytest.mark.parametrize(
    'hint, expected',
    [
        (int, None),
        (Ray, _DS(Ray, '...')),
        (Ray['h w'], _DS(Ray, 'h w')),
        (Ray[..., 3], _DS(Ray, '... 3')),
        (Optional[Ray], _DS(Ray, '...')),
        (Union[Ray, Camera], _DS(dca.DataclassArray, '...')),
        (Union[Ray, Camera, None], _DS(dca.DataclassArray, '...')),
        (Union[Ray, int], None),
        (Union[Ray, int, None], None),
        (Union[f32[3, 3], int, None], None),
        (List[int], None),
        (List[Ray], None),
        (f32[3, 3], f32[3, 3]),
        (FloatArray[..., 3], FloatArray[..., 3]),
    ],
)
def test_get_array_type(hint, expected):
  assert type_parsing.get_array_type(hint) == expected


@pytest.mark.parametrize(
    'hint, expected',
    [
        (Ray, _DS(Ray, '...')),
        (Ray['h w'], _DS(Ray, 'h w')),
        (Ray[..., 3], _DS(Ray, '... 3')),
    ],
)
def test_from_hint(hint, expected):
  assert dca.field_utils.DataclassWithShape.from_hint(hint) == expected


def test_get_array_type_error():
  with pytest.raises(NotImplementedError):
    type_parsing.get_array_type(Union[Ray, f32[3, 3]])

  with pytest.raises(NotImplementedError):
    type_parsing.get_array_type(Union[FloatArray[..., 3], f32[3, 3]])


@pytest.mark.parametrize(
    'hint, expected',
    [
        (
            Ray,
            dca.array_dataclass._ArrayFieldMetadata(
                inner_shape_non_static=(),
                dtype=Ray,
            ),
        ),
        (
            Ray[..., 3],
            dca.array_dataclass._ArrayFieldMetadata(
                inner_shape_non_static=(3,),
                dtype=Ray,
            ),
        ),
        (
            Ray['*shape 4 _'],
            dca.array_dataclass._ArrayFieldMetadata(
                inner_shape_non_static=(4, None),
                dtype=Ray,
            ),
        ),
    ],
)
def test_type_to_field_metadata(hint, expected):
  assert dca.array_dataclass._type_to_field_metadata(hint) == expected
