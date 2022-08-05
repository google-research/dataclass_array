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

import dataclasses
from typing import Optional, List, Union

import dataclass_array as dca
from dataclass_array import type_parsing
from etils.array_types import f32, FloatArray  # pylint: disable=g-multiple-import
import pytest


Ray = dca.testing.Ray


@dataclasses.dataclass(frozen=True)
class Camera(dca.DataclassArray):
  pos: FloatArray[..., 3]
  dir: FloatArray[..., 3]


@pytest.mark.parametrize(
    'hint, expected',
    [
        (int, [int]),
        (Ray, [Ray]),
        (Union[Ray, int], [Ray, int]),
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
        (Ray, Ray),
        (Optional[Ray], Ray),
        (Union[Ray, Camera], dca.DataclassArray),
        (Union[Ray, Camera, None], dca.DataclassArray),
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


def test_get_array_type_error():
  with pytest.raises(NotImplementedError):
    type_parsing.get_array_type(Union[Ray, f32[3, 3]])

  with pytest.raises(NotImplementedError):
    type_parsing.get_array_type(Union[FloatArray[..., 3], f32[3, 3]])
