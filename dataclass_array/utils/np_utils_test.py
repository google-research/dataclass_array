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

"""Tests for np_utils."""

from __future__ import annotations

import dataclass_array as dca
from dataclass_array.utils import np_utils
from etils import enp
import numpy as np
import pytest

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_get_xnp(xnp: enp.NpModule):
  # Dataclass array support
  r = dca.testing.Ray(
      pos=xnp.asarray([3.0, 0, 0]), dir=xnp.asarray([3.0, 0, 0])
  )
  assert np_utils.get_xnp(r) is xnp
  # Array support
  assert np_utils.get_xnp(xnp.asarray([3.0, 0, 0])) is xnp

  with pytest.raises(TypeError, match='Unexpected array type'):
    np_utils.get_xnp('abc')


def test_to_absolute_axis():
  assert np_utils.to_absolute_axis(None, ndim=4) == (0, 1, 2, 3)
  assert np_utils.to_absolute_axis(0, ndim=4) == 0
  assert np_utils.to_absolute_axis(1, ndim=4) == 1
  assert np_utils.to_absolute_axis(2, ndim=4) == 2
  assert np_utils.to_absolute_axis(3, ndim=4) == 3
  assert np_utils.to_absolute_axis(-1, ndim=4) == 3
  assert np_utils.to_absolute_axis(-2, ndim=4) == 2
  assert np_utils.to_absolute_axis(-3, ndim=4) == 1
  assert np_utils.to_absolute_axis(-4, ndim=4) == 0
  assert np_utils.to_absolute_axis((0,), ndim=4) == (0,)
  assert np_utils.to_absolute_axis((0, 1), ndim=4) == (0, 1)
  assert np_utils.to_absolute_axis((0, 1, -1), ndim=4) == (0, 1, 3)
  assert np_utils.to_absolute_axis((-1, -2), ndim=4) == (3, 2)

  with pytest.raises(np.AxisError):
    assert np_utils.to_absolute_axis(4, ndim=4)

  with pytest.raises(np.AxisError):
    assert np_utils.to_absolute_axis(-5, ndim=4)

  with pytest.raises(np.AxisError):
    assert np_utils.to_absolute_axis((0, 4), ndim=4)


def test_to_absolute_einops():
  assert (
      np_utils.to_absolute_einops(
          'b (h w) -> b h w',
          nlastdim=2,
      )
      == 'b (h w)  arr__0 arr__1 -> b h w arr__0 arr__1'
  )
  assert (
      np_utils.to_absolute_einops(
          'b (h w) arr__0 -> b h w arr__0',
          nlastdim=2,
      )
      == 'b (h w) arr__0  arr__1 arr__2 -> b h w arr__0 arr__1 arr__2'
  )
