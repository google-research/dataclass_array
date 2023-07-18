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

"""Tests for array_dataclass."""

from __future__ import annotations

import dataclass_array as dca
from dataclass_array.typing import FloatArray, IntArray, f32, i32  # pylint: disable=g-multiple-import
from etils import enp
import tensorflow as tf

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp

# TODO(epot): Test dtype `complex`, `str`


@dca.dataclass_array(broadcast=True, cast_dtype=True)
class Point(dca.DataclassArray):
  x: f32['*shape']
  y: f32['*shape']


def test_tf_data():
  p = Point(x=tf.constant(0), y=tf.constant(0))
  p = tf.nest.map_structure(lambda x: x + 1, p)
  assert p.x.numpy() == 1  # Works

  ds = tf.data.Dataset.range(3)
  ds = ds.map(lambda x: Point(x=x, y=x))  # Fail

  for ex in ds:
    assert isinstance(ex, Point)
    assert ex.x.shape == ()
    assert ex.y.shape == ()
