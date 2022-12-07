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

"""Test utils."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional

from dataclass_array import array_dataclass
from dataclass_array.typing import FloatArray  # pylint: disable=g-multiple-import
from etils import enp
from etils import epy
from etils.etree import jax as etree
from etils.etree import Tree
import numpy as np


@dataclasses.dataclass(frozen=True)
class Ray(array_dataclass.DataclassArray):
  """Dummy dataclass array for testing."""

  pos: FloatArray[..., 3]
  dir: FloatArray[..., 3]


# TODO(epot): Should use `chex.assert_xyz` once dataclasses support DM `tree`
def assert_trees(assert_fn, x: Tree[Any], y: Tree[Any]) -> None:
  """Compare all values."""
  etree.backend.assert_same_structure(x, y)
  # TODO(epot): Better error messages
  etree.backend.map(assert_fn, x, y)


def assert_allclose(
    x: Tree[Any],
    y: Tree[Any],
    *,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
  """Assert the trees are close."""
  kwargs = {}
  if atol is not None:
    kwargs['atol'] = atol
  if rtol is not None:
    kwargs['rtol'] = rtol
  assert_all_close_fn = functools.partial(np.testing.assert_allclose, **kwargs)
  assert_trees(assert_all_close_fn, x, y)


def assert_array_equal(
    x,
    y,
) -> None:
  """Assert the 2 objects are equals.

  * Support Dataclass arrays
  * Compare float arrays with allclose
  * Contrary to `np.testing.assert_allclose`, do not convert to array first.

  Args:
    x: First element to compare
    y: Second element to compare
  """
  assert type(x) == type(y)  # pylint: disable=unidiomatic-typecheck
  assert x.shape == y.shape
  assert_allclose(x, y)
  if isinstance(x, array_dataclass.DataclassArray):
    assert x.xnp is y.xnp


def assert_xnp(x: array_dataclass.DataclassArray, xnp: enp.NpModule) -> None:
  """Recursively check that all values are `xnp`."""
  assert isinstance(x, array_dataclass.DataclassArray)
  assert x.xnp is xnp
  # Do not use `etree` as it is meant to check that DataclassArray internals
  # are working correctly
  for k, v in x._all_array_fields.items():  # pylint: disable=protected-access
    v = v.value
    try:
      if v is None:
        continue
      elif enp.lazy.is_array(v):  # xnp array
        assert enp.lazy.get_xnp(v) is xnp
      elif isinstance(v, array_dataclass.DataclassArray):
        assert_xnp(v, xnp)
      else:
        raise TypeError(f'Unexpected value for {k}: {type(v)}')
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(e, prefix=f'{k}:')
