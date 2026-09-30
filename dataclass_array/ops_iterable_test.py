# Copyright 2026 The dataclass_array Authors.
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

"""Stack and concat must consume iterable inputs only once."""

from __future__ import annotations

import dataclass_array as dca
from dataclass_array.typing import FloatArray
from etils import enp
import numpy as np
import pytest

enable_tf_np_mode = enp.testing.set_tnp


class Point(dca.DataclassArray):
  position: FloatArray['*shape 3']
  mass: FloatArray['*shape']


class Samples(dca.DataclassArray):
  point: Point
  weight: FloatArray['*shape']
  label: str = 'sample'


def _samples(xnp):
  values = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
  return [
      Samples(
          point=Point(position=values + i, mass=values[..., 0] + i),
          weight=values[..., 1] + i,
      ).as_xnp(xnp)
      for i in range(3)
  ]


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('operation', [dca.stack, dca.concat])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('iterator_kind', ['iterator', 'generator'])
def test_single_pass_inputs_match_lists(xnp, operation, axis, iterator_kind):
  arrays = _samples(xnp)
  expected = operation(arrays, axis=axis)
  values = (
      iter(arrays)
      if iterator_kind == 'iterator'
      else (array for array in arrays)
  )
  actual = operation(values, axis=axis)
  dca.testing.assert_array_equal(actual, expected)
  assert actual.shape == expected.shape
  assert actual.xnp is expected.xnp
  assert actual.label == 'sample'


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('operation', [dca.stack, dca.concat])
def test_iterable_is_materialized_once(xnp, operation):
  arrays = _samples(xnp)

  class SinglePass:

    def __init__(self):
      self.iterations = 0

    def __iter__(self):
      self.iterations += 1
      if self.iterations > 1:
        raise AssertionError('Input iterable traversed more than once')
      yield from arrays

  values = SinglePass()
  actual = operation(values)
  dca.testing.assert_array_equal(actual, operation(arrays))
  assert values.iterations == 1
