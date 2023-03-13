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

"""Tests for vectorization."""

from __future__ import annotations

import dataclasses

import dataclass_array as dca
from dataclass_array import vectorization
from dataclass_array.typing import FloatArray
from dataclass_array.utils import inspect_utils
from dataclass_array.utils import np_utils
from etils import enp
import jax
import pytest
import tensorflow.experimental.numpy as tnp

H = 2
W = 3
X0 = 4
X1 = 5

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@pytest.mark.parametrize(
    [
        'self_shape',
        'arg_shape',
        'expected_arg_shape',
    ],
    [
        ((H,), (H,), (H,)),
        ((1,), (H,), (H,)),
        ((H,), (1,), (H,)),
        ((H,), (H, X0, X1), (H, X0, X1)),
        ((1,), (H, X0, X1), (H, X0, X1)),
        ((H,), (1, X0, X1), (H, X0, X1)),
        ((H, W), (H, W), (H * W,)),
        ((1, 1), (H, W), (H * W,)),
        ((H, W), (1, 1), (H * W,)),
        ((1, W), (H, 1), (H * W,)),
        ((H, W), (H, W, X0, X1), (H * W, X0, X1)),
        ((1, 1), (H, W, X0, X1), (H * W, X0, X1)),
        ((H, W), (1, 1, X0, X1), (H * W, X0, X1)),
        ((1, W), (H, 1, X0, X1), (H * W, X0, X1)),
    ],
)
@enp.testing.parametrize_xnp()
def test_broadcast_args(
    self_shape: dca.typing.Shape,
    arg_shape: dca.typing.Shape,
    expected_arg_shape: dca.typing.Shape,
    xnp: enp.NpModule,
):
  def fn(self, arg_dc, arg_array):
    assert isinstance(self, dca.testing.Ray)
    assert isinstance(arg_dc, dca.testing.Ray)
    assert enp.compat.is_array_xnp(arg_array, xnp)
    assert self.shape == ()  # pylint: disable=g-explicit-bool-comparison
    assert arg_dc.shape == expected_arg_shape[1:]
    assert arg_array.shape == expected_arg_shape[1:] + (3,)

  self = dca.testing.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  self = self.as_xnp(xnp)
  self = self.broadcast_to(self_shape)

  arg_dc = dca.testing.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  arg_dc = arg_dc.as_xnp(xnp)
  arg_dc = arg_dc.broadcast_to(arg_shape)

  arg_array = xnp.zeros(arg_shape + (3,))

  bound_args = inspect_utils.Signature(fn).bind(self, arg_dc, arg_array)
  bound_args, batch_shape = vectorization._broadcast_and_flatten_args(
      bound_args,
      map_non_static=lambda fn, args: args.map(fn),
  )

  assert len(bound_args) == 3
  new_self, new_dc, new_array = bound_args
  new_self = new_self.value
  new_dc = new_dc.value
  new_array = new_array.value

  # Self is flatten
  flat_batch_shape = (np_utils.size_of(batch_shape),)
  assert new_self.shape == flat_batch_shape
  assert expected_arg_shape[:1] == flat_batch_shape

  # Other are broadcasted to a self.flatten compatible size
  assert new_dc.shape == expected_arg_shape
  assert new_array.shape == expected_arg_shape + (3,)


@pytest.mark.parametrize(
    [
        'self_shape',
        'arg_shape',
    ],
    [
        ((H, W), ()),
        ((H, W), (H,)),
        ((H, W), (W,)),
        ((H, W), (H, X0)),
        ((H, W), (X0, W)),
    ],
)
@enp.testing.parametrize_xnp()
def test_broadcast_args_failure(
    self_shape: dca.typing.Shape,
    arg_shape: dca.typing.Shape,
    xnp: enp.NpModule,
):
  def fn(self, arg):
    del self, arg

  self = dca.testing.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  self = self.as_xnp(xnp)
  self = self.broadcast_to(self_shape)

  arg_dc = dca.testing.Ray(pos=[0, 0, 0], dir=[0, 0, 0])
  arg_dc = arg_dc.as_xnp(xnp)
  arg_dc = arg_dc.broadcast_to(arg_shape)

  bound_args = inspect_utils.Signature(fn).bind(self, arg_dc)

  with pytest.raises(ValueError, match='Cannot vectorize shape'):
    vectorization._broadcast_and_flatten_args(
        bound_args,
        map_non_static=lambda fn, args: args.map(fn),
    )


@enp.testing.parametrize_xnp()
def test_replace_dca(xnp: enp.NpModule):
  # Ensure that the non-init static fields are correctly forwarded.

  class DataclassWithNonInit(dca.DataclassArray):
    """Dataclass with a non-init (static) field."""

    __dca_non_init_fields__ = ('x',)

    y: FloatArray['*batch']
    x: int = dataclasses.field(default=1, init=False)

    @dca.vectorize_method
    def fn(self):
      assert not self.shape
      assert self.x == 5
      return self

  a = DataclassWithNonInit(y=[1, 0, 0]).as_xnp(xnp)
  assert a.shape == (3,)
  assert a.x == 1

  # Replace supported
  a = a.replace(x=5)
  assert a.shape == (3,)
  assert a.x == 5

  a = a.replace(y=a.y + 1)
  assert a.shape == (3,)
  assert a.x == 5

  # Vectorization supported
  if xnp not in [
      tnp,
  ]:
    a = a.fn()
  assert a.xnp is xnp
  assert a.shape == (3,)
  assert a.x == 5

  # Tree-map supported
  a = jax.tree_util.tree_map(lambda x: x, a)
  assert a.shape == (3,)
  assert a.x == 5
