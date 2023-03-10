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

"""Tests for array_dataclass."""

from __future__ import annotations

from typing import Any

import dataclass_array as dca
from dataclass_array.typing import FloatArray, IntArray, f32, i32  # pylint: disable=g-multiple-import
from dataclass_array.typing import Shape  # pylint: disable=g-multiple-import
from etils import enp
import numpy as np
import pytest
import tensorflow as tf

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp

# TODO(epot): Test dtype `complex`, `str`


class DcaTest:

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> dca.DataclassArray:
    """Construct the dataclass array with the given shape."""
    raise NotImplementedError

  @staticmethod
  def assert_val(
      p: dca.DataclassArray, shape: Shape, xnp: enp.NpModule = None
  ) -> None:
    """Test that the value has expected shape/xnp."""
    raise NotImplementedError


@dca.dataclass_array(broadcast=True, cast_dtype=True)
class Point(dca.DataclassArray):
  x: f32['*shape']
  y: f32['*shape']

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> Point:
    """Construct the dataclass array with the given shape."""
    return Point(
        x=xnp.zeros(shape),
        y=xnp.zeros(shape),
    )

  @staticmethod
  def assert_val(p: Point, shape: Shape, xnp: enp.NpModule = None):
    """Validate the point."""
    xnp = xnp or np
    assert isinstance(p, Point)
    _assert_common(p, shape=shape, xnp=xnp)
    assert p.x.shape == shape
    assert p.y.shape == shape
    assert enp.lazy.as_dtype(p.x.dtype) == np.float32
    assert enp.lazy.as_dtype(p.y.dtype) == np.float32
    assert enp.compat.is_array_xnp(p.x, xnp)
    assert enp.compat.is_array_xnp(p.y, xnp)


@dca.dataclass_array(broadcast=True, cast_dtype=True)
class Isometrie(dca.DataclassArray):
  r: f32['... 3 3']
  t: i32[..., 2]

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> Isometrie:
    """Construct the dataclass array with the given shape."""
    return Isometrie(
        r=xnp.zeros(shape + (3, 3)),
        t=xnp.zeros(shape + (2,)),
    )

  @staticmethod
  def assert_val(p: Isometrie, shape: Shape, xnp: enp.NpModule = None):
    """Validate the point."""
    xnp = xnp or np
    assert isinstance(p, Isometrie)
    _assert_common(p, shape=shape, xnp=xnp)
    assert p.r.shape == shape + (3, 3)
    assert p.t.shape == shape + (2,)
    assert enp.lazy.as_dtype(p.r.dtype) == np.float32
    assert enp.lazy.as_dtype(p.t.dtype) == np.int32
    assert enp.compat.is_array_xnp(p.r, xnp)
    assert enp.compat.is_array_xnp(p.t, xnp)


@dca.dataclass_array(broadcast=True, cast_dtype=True)
class Nested(dca.DataclassArray):
  # pytype: disable=annotation-type-mismatch
  iso: Isometrie
  iso_batched: Isometrie['*batch_shape 3 7']
  pt: Point = dca.field(shape=(3,), dtype=Point)
  # pytype: enable=annotation-type-mismatch

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> Nested:
    """Construct the dataclass array with the given shape."""
    return Nested(
        pt=Point(
            x=xnp.zeros(shape + (3,)),
            y=xnp.zeros(shape + (3,)),
        ),
        iso=Isometrie(
            r=xnp.zeros(shape + (3, 3)),
            t=xnp.zeros(shape + (2,)),
        ),
        iso_batched=Isometrie(
            r=xnp.zeros(shape + (3, 7, 3, 3)),
            t=xnp.zeros(shape + (3, 7, 2)),
        ),
    )

  @staticmethod
  def assert_val(p: Nested, shape: Shape, xnp: enp.NpModule = None):
    """Validate the nested."""
    xnp = xnp or np
    assert isinstance(p, Nested)
    _assert_common(p, shape=shape, xnp=xnp)
    Point.assert_val(p.pt, shape=shape + (3,), xnp=xnp)
    Isometrie.assert_val(p.iso, shape=shape, xnp=xnp)
    Isometrie.assert_val(p.iso_batched, shape=shape + (3, 7), xnp=xnp)


class OnlyStatic(dca.DataclassArray):
  """Dataclass with no array fields."""

  x: int
  y: int

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> OnlyStatic:
    """Construct the dataclass array with the given shape."""

    val = OnlyStatic(
        x=0,
        y=1,
    )
    assert val.shape == ()  # pylint: disable=g-explicit-bool-comparison
    assert val.xnp == np

    return val.broadcast_to(shape).as_xnp(xnp)

  @staticmethod
  def assert_val(p: OnlyStatic, shape: Shape, xnp: enp.NpModule = None):
    """Validate the point."""
    xnp = xnp or np
    assert isinstance(p, OnlyStatic)
    _assert_common(p, shape=shape, xnp=xnp)
    assert p.x == 0
    assert p.y == 1


class NestedOnlyStatic(dca.DataclassArray):
  """Dataclass with only nested array fields."""

  s0: OnlyStatic
  s1: OnlyStatic

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> NestedOnlyStatic:
    """Construct the dataclass array with the given shape."""
    return NestedOnlyStatic(
        s0=OnlyStatic(x=0, y=1).as_xnp(xnp).broadcast_to(shape),
        s1=OnlyStatic(x=0, y=1).broadcast_to(shape),
    )

  @staticmethod
  def assert_val(p: NestedOnlyStatic, shape: Shape, xnp: enp.NpModule = None):
    """Validate the point."""
    xnp = xnp or np
    assert isinstance(p, NestedOnlyStatic)
    _assert_common(p, shape=shape, xnp=xnp)
    OnlyStatic.assert_val(p.s0, shape=shape, xnp=xnp)
    OnlyStatic.assert_val(p.s1, shape=shape, xnp=xnp)


@dca.dataclass_array(broadcast=True, cast_dtype=True)
class WithStatic(dca.DataclassArray):
  """Mix of static and array fields."""

  static: str
  nested_static: NestedOnlyStatic
  x: f32['... 3']
  y: Any = dca.field(shape=(2, 2), dtype=np.float32)

  @staticmethod
  def make(shape: Shape, xnp: enp.NpModule) -> WithStatic:
    """Construct the dataclass array with the given shape."""
    return WithStatic(
        nested_static=NestedOnlyStatic.make(shape, xnp),
        x=xnp.zeros(shape + (3,)),
        y=xnp.zeros(shape + (2, 2)),
        static='abc',
    )

  @staticmethod
  def assert_val(p: WithStatic, shape: Shape, xnp: enp.NpModule = None):
    """Validate the point."""
    xnp = xnp or np
    assert isinstance(p, WithStatic)
    _assert_common(p, shape=shape, xnp=xnp)
    NestedOnlyStatic.assert_val(p.nested_static, shape, xnp=xnp)
    assert p.x.shape == shape + (3,)
    assert p.y.shape == shape + (2, 2)
    assert enp.lazy.as_dtype(p.x.dtype) == np.float32
    assert enp.lazy.as_dtype(p.y.dtype) == np.float32
    assert enp.compat.is_array_xnp(p.x, xnp)
    assert enp.compat.is_array_xnp(p.y, xnp)
    # Static field is correctly forwarded
    assert isinstance(p.static, str)
    assert p.static == 'abc'


def _assert_common(p: dca.DataclassArray, shape: Shape, xnp: enp.NpModule):
  """Test the len(p)."""
  assert p  # Object evaluate to True
  assert p.shape == shape

  p_np = np.empty(shape)
  assert p.size == p_np.size
  assert p.ndim == p_np.ndim
  assert p.xnp is xnp
  if shape:
    assert len(p) == shape[0]
  else:
    with pytest.raises(TypeError, match='of unsized '):
      _ = len(p)


parametrize_dataclass_arrays = pytest.mark.parametrize(
    'dca_cls',
    [
        Point,
        Isometrie,
        Nested,
        OnlyStatic,
        NestedOnlyStatic,
        WithStatic,
    ],
)


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize(
    'x, y, shape',
    [
        (1, 2, ()),
        ([1, 2], [3, 4], (2,)),
        ([[1], [2]], [[3], [4]], (2, 1)),
    ],
)
def test_point_infered_np(
    xnp: enp.NpModule,
    x,
    y,
    shape: Shape,
):
  if xnp is not None:  # Normalize np arrays to test the various backend
    x = xnp.asarray(x)
    y = xnp.asarray(y)
  else:
    xnp = np

  p = Point(x=x, y=y)
  Point.assert_val(p, shape, xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_scalar_shape(
    xnp: enp.NpModule,
    dca_cls: DcaTest,
):
  p = dca_cls.make(shape=(), xnp=xnp)
  dca_cls.assert_val(p, (), xnp=xnp)
  dca_cls.assert_val(p.replace(), (), xnp=xnp)
  dca_cls.assert_val(p.broadcast_to((2,)).replace(), (2,), xnp=xnp)
  dca_cls.assert_val(p.reshape((1, 1, 1)), (1, 1, 1), xnp=xnp)
  dca_cls.assert_val(p.flatten(), (1,), xnp=xnp)
  dca_cls.assert_val(p.broadcast_to((1, 3, 2)), (1, 3, 2), xnp=xnp)

  with pytest.raises(TypeError, match='iteration over'):
    _ = list(p)

  with pytest.raises(IndexError, match='too many indices for array'):
    _ = p[0]

  dca_cls.assert_val(p[...], (), xnp=xnp)  # Index on ... is a no-op

  dca_cls.assert_val(dca.stack([p, p, p]), (3,), xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_simple_shape(
    xnp: enp.NpModule,
    dca_cls: DcaTest,
):
  p = dca_cls.make(shape=(3, 2), xnp=xnp)
  dca_cls.assert_val(p, (3, 2), xnp=xnp)
  dca_cls.assert_val(p.reshape((2, 1, 3, 1, 1)), (2, 1, 3, 1, 1), xnp=xnp)
  dca_cls.assert_val(p.flatten(), (6,), xnp=xnp)
  dca_cls.assert_val(p.broadcast_to((7, 4, 3, 2)), (7, 4, 3, 2), xnp=xnp)
  dca_cls.assert_val(p[0], (2,), xnp=xnp)
  dca_cls.assert_val(p[1, 1], (), xnp=xnp)
  dca_cls.assert_val(p[:, 0], (3,), xnp=xnp)
  dca_cls.assert_val(p[..., 0], (3,), xnp=xnp)
  dca_cls.assert_val(p[0, ...], (2,), xnp=xnp)
  dca_cls.assert_val(p[...], (3, 2), xnp=xnp)

  p0, p1, p2 = list(p)
  dca_cls.assert_val(p0, (2,), xnp=xnp)
  dca_cls.assert_val(p1, (2,), xnp=xnp)
  dca_cls.assert_val(p2, (2,), xnp=xnp)

  dca_cls.assert_val(dca.stack([p0, p0, p1, p1]), (4, 2), xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_complex_shape(
    xnp: enp.NpModule,
    dca_cls: DcaTest,
):
  p = dca_cls.make(shape=(3, 2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p, (3, 2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p.reshape((2, 1, 3, 1, 1)), (2, 1, 3, 1, 1), xnp=xnp)
  dca_cls.assert_val(p.flatten(), (6,), xnp=xnp)
  dca_cls.assert_val(p.broadcast_to((7, 3, 2, 1, 1)), (7, 3, 2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p[0], (2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p[1, 1], (1, 1), xnp=xnp)
  dca_cls.assert_val(p[:, 0], (3, 1, 1), xnp=xnp)
  dca_cls.assert_val(p[:, 0, 0, :], (3, 1), xnp=xnp)
  dca_cls.assert_val(p[..., 0], (3, 2, 1), xnp=xnp)
  dca_cls.assert_val(p[0, ...], (2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p[0, ..., 0], (2, 1), xnp=xnp)
  # Indexing through np.array also supported
  dca_cls.assert_val(
      p.flatten()[np.ones(p.size, dtype=np.bool_)],
      (p.size,),
      xnp=xnp,
  )
  dca_cls.assert_val(
      p.flatten()[xnp.ones(p.size, dtype=enp.lazy.as_dtype(np.bool_, xnp=xnp))],
      (p.size,),
      xnp=xnp,
  )

  p0, p1, p2 = list(p)
  dca_cls.assert_val(p0, (2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p1, (2, 1, 1), xnp=xnp)
  dca_cls.assert_val(p2, (2, 1, 1), xnp=xnp)

  dca_cls.assert_val(dca.stack([p0, p0, p1, p1]), (4, 2, 1, 1), xnp=xnp)


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_einops_reshape(
    xnp: enp.NpModule,
    dca_cls: DcaTest,
):
  p = dca_cls.make(shape=(1, 2, 4), xnp=xnp)

  dca_cls.assert_val(p, (1, 2, 4), xnp=xnp)
  dca_cls.assert_val(p.reshape('b h w -> b (h w)'), (1, 2 * 4), xnp=xnp)
  dca_cls.assert_val(
      p.reshape('b h (w0 w1) -> b w0 h w1', w0=2, w1=2),
      (1, 2, 2, 2),
      xnp=xnp,
  )


def test_wrong_input_type():
  pts = Point(x=1, y=1)

  # str instead of ndarray
  with pytest.raises(TypeError, match='Could not infer numpy module'):
    _ = Point(
        x=1,
        y='a',
    )

  # DataclassArray instead of ndarray
  with pytest.raises(TypeError, match='Invalid Point.y: Expected .*float32'):
    _ = Point(
        x=1,
        y=pts,
    )

  @dca.dataclass_array(broadcast=True, cast_dtype=True)
  class PointWrapper(dca.DataclassArray):
    pts: Point
    rgb: f32['*shape 3']

  # ndarray instead of DataclassArray
  with pytest.raises(TypeError, match='Invalid PointWrapper.pts:'):
    _ = PointWrapper(
        pts=enp.lazy.np.array(0.1),
        rgb=pts,
    )

  # TODO(epot): Test invalid casting (float -> int)
  # TODO(epot): Test non-matching shape


def test_isometrie_wrong_input():
  # Incompatible types
  with pytest.raises(ValueError, match='Conflicting numpy types'):
    # TODO(epot): Could also test with `.replace()`, and check that nested
    # static behave as expected
    _ = Isometrie(
        r=enp.lazy.jnp.zeros((3, 3)),
        t=enp.lazy.tnp.zeros((2,)),
    )

  # Bad inner shape
  with pytest.raises(ValueError, match='Invalid Isometrie.r: Shape do not'):
    _ = Isometrie(
        r=np.zeros((3, 2)),
        t=np.zeros((2,)),
    )

  # Bad batch shape
  with pytest.raises(ValueError, match='Conflicting batch shapes'):
    _ = Isometrie(
        r=np.zeros((2, 3, 3)),
        t=np.zeros((3, 2)),
    )

  # Bad reshape
  p = Isometrie(
      r=np.zeros((3, 3, 3)),
      t=np.zeros((3, 2)),
  )
  with pytest.raises(ValueError, match='cannot reshape array'):
    p.reshape((2, 2))


@pytest.mark.parametrize(
    'batch_shape, indices',
    [
        ((), np.index_exp[...]),
        ((2,), np.index_exp[...]),
        ((3, 2), np.index_exp[...]),
        ((3, 2), np.index_exp[0]),
        ((3, 2), np.index_exp[0, ...]),
        ((3, 2), np.index_exp[..., 0]),
        ((3, 2), np.index_exp[0, 0]),
        ((3, 2), np.index_exp[..., 0, 0]),
        ((3, 2), np.index_exp[0, ..., 0]),
        ((3, 2), np.index_exp[0, 0, ...]),
        ((3, 2), np.index_exp[0, :, ...]),
        ((3, 2), np.index_exp[:, ..., :]),
        ((3, 2), np.index_exp[None,]),
        ((3, 2), np.index_exp[None, :]),
        ((3, 2), np.index_exp[np.newaxis, :]),
        ((2,), np.index_exp[None, ..., None, 0, None, None]),
        ((2,), np.index_exp[None, ..., None, 0, None, None]),
        ((3, 2), np.index_exp[None, ..., None, 0, None, None]),
        ((3, 1, 2), np.index_exp[None, ..., None, 0, None, None]),
    ],
)
def test_normalize_indices(batch_shape: Shape, indices):
  # Compare the indexing with and without the extra batch shcape
  x0 = np.ones(batch_shape + (4, 2))
  x1 = np.ones(batch_shape)

  normalized_indices = dca.array_dataclass._to_absolute_indices(
      indices,
      shape=batch_shape,
  )
  x0 = x0[normalized_indices]
  x1 = x1[indices]
  assert x0.shape == x1.shape + (4, 2)


@enp.testing.parametrize_xnp()
def test_empty(xnp: enp.NpModule):
  p = Point(x=xnp.empty((0, 3)), y=xnp.empty((0, 3)))  # Empty array

  with pytest.raises(ValueError, match='The truth value of'):
    bool(p)


@enp.testing.parametrize_xnp()
def test_absolute_axis(xnp: enp.NpModule):
  p = Isometrie(r=xnp.ones((3, 3)), t=xnp.ones((2,)))
  p = p.broadcast_to((1, 2, 3, 4))
  assert p.shape == (1, 2, 3, 4)

  assert p._to_absolute_axis(None) == (0, 1, 2, 3)
  assert p._to_absolute_axis(0) == 0
  assert p._to_absolute_axis(1) == 1
  assert p._to_absolute_axis(2) == 2
  assert p._to_absolute_axis(3) == 3
  assert p._to_absolute_axis(-1) == 3
  assert p._to_absolute_axis(-2) == 2
  assert p._to_absolute_axis(-3) == 1
  assert p._to_absolute_axis(-4) == 0
  assert p._to_absolute_axis((0,)) == (0,)
  assert p._to_absolute_axis((0, 1)) == (0, 1)
  assert p._to_absolute_axis((0, 1, -1)) == (0, 1, 3)
  assert p._to_absolute_axis((-1, -2)) == (3, 2)

  with pytest.raises(np.AxisError):
    assert p._to_absolute_axis(4)

  with pytest.raises(np.AxisError):
    assert p._to_absolute_axis(-5)

  with pytest.raises(np.AxisError):
    assert p._to_absolute_axis((0, 4))


@enp.testing.parametrize_xnp()
@parametrize_dataclass_arrays
def test_convert(
    xnp: enp.NpModule,
    dca_cls: DcaTest,
):
  p = dca_cls.make(xnp=xnp, shape=(2,))
  assert p.xnp is xnp

  assert p.as_np().xnp is enp.lazy.np
  assert p.as_jax().xnp is enp.lazy.jnp
  assert p.as_tf().xnp is enp.lazy.tnp
  assert p.as_torch().xnp is enp.lazy.torch
  assert p.as_xnp(np).xnp is enp.lazy.np
  assert p.as_xnp(enp.lazy.jnp).xnp is enp.lazy.jnp
  assert p.as_xnp(enp.lazy.tnp).xnp is enp.lazy.tnp
  assert p.as_xnp(enp.lazy.torch).xnp is enp.lazy.torch
  # Make sure the nested class are also updated
  dca_cls.assert_val(p.as_jax(), (2,), xnp=enp.lazy.jnp)


@enp.testing.parametrize_xnp()
def test_broadcast(xnp: enp.NpModule):
  broadcast_shape = (3, 2)
  p = Nested(
      # pt.shape broadcasted to (2, 3)
      pt=Point(
          x=xnp.asarray(0),
          y=xnp.zeros(broadcast_shape + (3,)),
      ),
      # iso.shape == (), broadcasted to (2, 3)
      iso=Isometrie(
          r=xnp.zeros((3, 3)),
          t=xnp.zeros((2,)),
      ),
      # iso.shape == (2, 3) + (3, 7) + (2,)
      iso_batched=Isometrie(
          r=xnp.zeros((3, 3)),
          t=xnp.zeros((3, 7) + (2,)),
      ),
  )
  Nested.assert_val(p, shape=broadcast_shape, xnp=xnp)


@enp.testing.parametrize_xnp()
def test_infer_np(xnp: enp.NpModule):
  p = Point(x=xnp.ones((3,)), y=[0, 0, 0])  # y is casted to xnp
  assert p.xnp is xnp
  assert enp.compat.is_array_xnp(p.y, xnp)


@parametrize_dataclass_arrays
@pytest.mark.parametrize(
    'tree_map',
    [
        enp.lazy.jax.tree_map,
        enp.lazy.torch.utils._pytree.tree_map,
    ],
)
def test_tree_map(tree_map, dca_cls: DcaTest):
  p = dca_cls.make(shape=(3,), xnp=np)
  p = tree_map(lambda x: x[None, ...], p)
  dca_cls.assert_val(p, (1, 3), xnp=np)


def test_torch_device():
  p = Nested.make(shape=(2,), xnp=enp.lazy.torch)
  p = p.cpu()
  p = p.to('cpu')
  Nested.assert_val(p, (2,), xnp=enp.lazy.torch)


@enp.testing.parametrize_xnp(
    restrict=[
        'jnp',
        'torch',
    ]
)
def test_vmap(xnp: enp.NpModule):
  import functorch

  vmap_fn = {
      enp.lazy.jnp: enp.lazy.jax.vmap,
      enp.lazy.torch: functorch.vmap,
  }[xnp]

  batch_shape = 3

  @vmap_fn
  def fn(p: WithStatic) -> WithStatic:
    assert isinstance(p, WithStatic)
    assert p.shape == ()  # pylint:disable=g-explicit-bool-comparison
    return p.replace(x=p.x + 1)

  x = WithStatic.make((batch_shape,), xnp=xnp)
  y = fn(x)
  WithStatic.assert_val(y, (batch_shape,), xnp=xnp)
  # pos was updated
  np.testing.assert_allclose(y.x, np.ones((batch_shape, 3)))
  np.testing.assert_allclose(y.y, np.zeros((batch_shape, 2, 2)))


@enp.testing.parametrize_xnp()
def test_dataclass_params_no_cast(xnp: enp.NpModule):
  class PointNoCast(dca.DataclassArray):
    x: FloatArray['*shape']
    y: IntArray['*shape']

  with pytest.raises(ValueError, match='Cannot cast float16'):
    PointNoCast(
        x=xnp.asarray([1, 2, 3], dtype=xnp.float16),
        y=xnp.asarray([1, 2, 3], dtype=xnp.float16),
    )

  p = PointNoCast(
      x=xnp.asarray([1, 2, 3], dtype=xnp.float16),
      y=xnp.asarray([1, 2, 3], dtype=xnp.uint8),
  )
  assert p.shape == (3,)
  assert enp.lazy.as_dtype(p.x.dtype) == np.float16
  assert enp.lazy.as_dtype(p.y.dtype) == np.uint8


@enp.testing.parametrize_xnp()
def test_dataclass_params_no_list(xnp: enp.NpModule):
  @dca.dataclass_array(cast_list=False)
  class PointNoList(dca.DataclassArray):
    x: FloatArray['*shape']
    y: IntArray['*shape']

  with pytest.raises(TypeError, match='Could not infer numpy module'):
    PointNoList(
        x=xnp.asarray(1, dtype=xnp.float16),
        y=[1, 2, 3],
    )


@enp.testing.parametrize_xnp()
def test_dataclass_params_no_broadcast(xnp: enp.NpModule):
  class PointNoBroadcast(dca.DataclassArray):
    x: FloatArray['*shape']
    y: IntArray['*shape']

  with pytest.raises(ValueError, match='Cannot broadcast'):
    PointNoBroadcast(
        x=xnp.asarray(1, dtype=xnp.float16),
        y=xnp.asarray([1, 2, 3], dtype=xnp.int32),
    )


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('batch_shape', [(), (1, 3)])
def test_dataclass_none_shape(xnp: enp.NpModule, batch_shape: Shape):
  class PointDynamicShape(dca.DataclassArray):
    x: FloatArray[..., None, None]
    y: IntArray['... 3 _']

  p = PointDynamicShape(
      x=xnp.zeros(batch_shape + (2, 3), dtype=xnp.float32),
      y=xnp.zeros(batch_shape + (3, 1), dtype=xnp.int32),
  )
  assert p.shape == batch_shape
  assert p.x.shape == batch_shape + (2, 3)
  assert p.y.shape == batch_shape + (3, 1)

  p2 = PointDynamicShape(
      x=xnp.zeros(batch_shape + (3, 2), dtype=xnp.float32),
      y=xnp.zeros(batch_shape + (3, 1), dtype=xnp.int32),
  )
  assert p2.shape == batch_shape
  assert p2.x.shape == batch_shape + (3, 2)
  assert p2.y.shape == batch_shape + (3, 1)

  # We can stack when the shape is compatible
  assert dca.stack([p, p]).shape == (2,) + batch_shape

  # Incompatible shape will raise an error
  expected_exception_cls = {
      enp.lazy.np: ValueError,
      enp.lazy.jnp: ValueError,
      enp.lazy.tnp: tf.errors.InvalidArgumentError,
      enp.lazy.torch: RuntimeError,
  }
  with pytest.raises(expected_exception_cls[xnp]):
    dca.stack([p, p2])

  if batch_shape:
    err_msg = 'Conflicting batch shapes'
  else:
    err_msg = 'Shape do not match.'
  with pytest.raises(ValueError, match=err_msg):
    PointDynamicShape(
        x=xnp.zeros(batch_shape + (3,), dtype=xnp.float32),  # len() != 2
        y=xnp.zeros(batch_shape + (3, 1), dtype=xnp.int32),
    )

  with pytest.raises(
      ValueError,
      match='Shape do not match.',
  ):
    PointDynamicShape(
        x=xnp.zeros(batch_shape + (2, 3), dtype=xnp.float32),
        y=xnp.zeros(batch_shape + (2, 1), dtype=xnp.int32),  # < 2 != 3
    )


def test_class_getitem():
  assert Point == Point  # pylint: disable=comparison-with-itself
  assert Point[''] == Point['']
  assert Point['h w'] == Point['h w']
  assert Point[''] != Point
  assert Point[''] != Point['h w']
  assert Point[''] != Isometrie['']
