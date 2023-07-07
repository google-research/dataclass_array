# Dataclass Array

[![Unittests](https://github.com/google-research/dataclass_array/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/dataclass_array/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/dataclass_array.svg)](https://badge.fury.io/py/dataclass_array)
[![Documentation Status](https://readthedocs.org/projects/dataclass-array/badge/?version=latest)](https://dataclass-array.readthedocs.io/en/latest/?badge=latest)


`DataclassArray` are dataclasses which behave like numpy-like arrays (can be
batched, reshaped, sliced,...), compatible with Jax, TensorFlow, and numpy (with
torch support planned).

This reduce boilerplate and improve readability. See the
[motivating examples](#motivating-examples) section bellow.

To view an example of dataclass arrays used in practice, see
[visu3d](https://github.com/google-research/visu3d).

## Documentation

### Definition

To create a `dca.DataclassArray`, take a frozen dataclass and:

*   Inherit from `dca.DataclassArray`
*   Annotate the fields with `dataclass_array.typing` to specify the inner shape
    and dtype of the array (see below for static or nested dataclass fields).
    The array types are an alias from
    [`etils.array_types`](https://github.com/google/etils/blob/main/etils/array_types/README.md).

```python
import dataclass_array as dca
from dataclass_array.typing import FloatArray


class Ray(dca.DataclassArray):
  pos: FloatArray['*batch_shape 3']
  dir: FloatArray['*batch_shape 3']
```

### Usage

Afterwards, the dataclass can be used as a numpy array:

```python
ray = Ray(pos=jnp.zeros((3, 3)), dir=jnp.eye(3))


ray.shape == (3,)  # 3 rays batched together
ray.pos.shape == (3, 3)  # Individual fields still available

# Numpy slicing/indexing/masking
ray = ray[..., 1:2]
ray = ray[norm(ray.dir) > 1e-7]

# Shape transformation
ray = ray.reshape((1, 3))
ray = ray.reshape('h w -> w h')  # Native einops support
ray = ray.flatten()

# Stack multiple dataclass arrays together
ray = dca.stack([ray0, ray1, ...])

# Supports TF, Jax, Numpy (torch planned) and can be easily converted
ray = ray.as_jax()  # as_np(), as_tf()
ray.xnp == jax.numpy  # `numpy`, `jax.numpy`, `tf.experimental.numpy`

# Compatibility `with jax.tree_util`, `jax.vmap`,..
ray = jax.tree_util.tree_map(lambda x: x+1, ray)
```

A `DataclassArray` has 2 types of fields:

*   Array fields: Fields batched like numpy arrays, with reshape, slicing,...
    Can be `xnp.ndarray` or nested `dca.DataclassArray`.
*   Static fields: Other non-numpy field. Are not modified by reshaping,...
    Static fields are also ignored in `jax.tree_map`.

```python
class MyArray(dca.DataclassArray):
  # Array fields
  a: FloatArray['*batch_shape 3']  # Defined by `etils.array_types`
  b: FloatArray['*batch_shape _ _']  # Dynamic shape
  c: Ray  # Nested DataclassArray (equivalent to `Ray['*batch_shape']`)
  d: Ray['*batch_shape 6']

  # Array fields explicitly defined
  e: Any = dca.field(shape=(3,), dtype=np.float32)
  f: Any = dca.field(shape=(None,  None), dtype=np.float32)  # Dynamic shape
  g: Ray = dca.field(shape=(3,), dtype=Ray)  # Nested DataclassArray

  # Static field (everything not defined as above)
  static0: float
  static1: np.array
```

### Vectorization

`@dca.vectorize_method` allow your dataclass method to automatically support
batching:

1.  Implement method as if `self.shape == ()`
2.  Decorate the method with `dca.vectorize_method`

```python
class Camera(dca.DataclassArray):
  K: FloatArray['*batch_shape 4 4']
  resolution = tuple[int, int]

  @dca.vectorize_method
  def rays(self) -> Ray:
    # Inside `@dca.vectorize_method` shape is always guarantee to be `()`
    assert self.shape == ()
    assert self.K.shape == (4, 4)

    # Compute the ray as if there was only a single camera
    return Ray(pos=..., dir=...)
```

Afterward, we can generate rays for multiple camera batched together:

```python
cams = Camera(K=K)  # K.shape == (num_cams, 4, 4)
rays = cams.rays()  # Generate the rays for all the cameras

cams.shape == (num_cams,)
rays.shape == (num_cams, h, w)
```

`@dca.vectorize_method` is similar to `jax.vmap` but:

*   Only work on `dca.DataclassArray` methods
*   Instead of vectorizing a single axis, `@dca.vectorize_method` will vectorize
    over `*self.shape` (not just `self.shape[0]`). This is like if `vmap` was
    applied to `self.flatten()`
*   When multiple arguments, axis with dimension `1` are broadcasted.

For example, with `__matmul__(self, x: T) -> T`:

```python
() @ (*x,) -> (*x,)
(b,) @ (b, *x) -> (b, *x)
(b,) @ (1, *x) -> (b, *x)
(1,) @ (b, *x) -> (b, *x)
(b, h, w) @ (b, h, w, *x) -> (b, h, w, *x)
(1, h, w) @ (b, 1, 1, *x) -> (b, h, w, *x)
(a, *x) @ (b, *x) -> Error: Incompatible a != b
```

To test on Colab, see the `visu3d` dataclass
[Colab tutorial](https://colab.research.google.com/github/google-research/visu3d/blob/main/docs/dataclass.ipynb).

## Motivating examples

`dca.DataclassArray` improve readability by simplifying common patterns:

*   Reshaping all fields of a dataclass:

    Before (`rays` is simple `dataclass`):

    ```python
    num_rays = math.prod(rays.origins.shape[:-1])
    rays = jax.tree_map(lambda r: r.reshape((num_rays, -1)), rays)
    ```

    After (`rays` is `DataclassArray`):

    ```python
    rays = rays.flatten()  # (b, h, w) -> (b*h*w,)
    ```

*   Rendering a video:

    Before (`cams: list[Camera]`):

    ```python
    img = cams[0].render(scene)
    imgs = np.stack([cam.render(scene) for cam in cams[::2]])
    imgs = np.stack([cam.render(scene) for cam in cams])
    ```

    After (`cams: Camera` with `cams.shape == (num_cams,)`):

    ```python
    img = cams[0].render(scene)  # Render only the first camera (to debug)
    imgs = cams[::2].render(scene)  # Render 1/2 frames (for quicker iteration)
    imgs = cams.render(scene)  # Render all cameras at once
    ```

## Installation

```sh
pip install dataclass_array
```

*This is not an official Google product*
