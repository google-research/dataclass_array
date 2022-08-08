# Dataclass Array

`DataclassArray` are dataclasses which behave like numpy-like arrays (can be
batched, reshaped, sliced,...), but are compatible with Jax, TensorFlow, and
numpy (with torch support planned).

## Documentation

To create a `dca.DataclassArray`, take a frozen dataclass and:

*   Inherit from `dca.DataclassArray`
*   Annotate the fields with `etils.array_types` to specify the inner shape and
    dtype of the array (see below for static or nested dataclass fields).

```python
import dataclass_array as dca
from etils.array_types import FloatArray


@dataclasses.dataclass(frozen=True)
class Ray(dca.DataclassArray):
  pos: FloatArray['*batch_shape 3']
  dir: FloatArray['*batch_shape 3']
```

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
@dataclasses.dataclass(frozen=True)
class MyArray(dca.DataclassArray):
  # Array fields
  a: FloatArray['*batch_shape 3']  # Defined by `etils.array_types`
  b: Ray  # Nested DataclassArray (inner shape == `()`)

  # Array fields explicitly defined
  c: Any = dca.field(shape=(3,), dtype=np.float32)
  d: Ray = dca.field(shape=(3,), dtype=Ray)  # Nested DataclassArray

  # Static field (everything not defined as above)
  e: float
  f: np.array
```

## Installation

```sh
pip install dataclass_array
```

*This is not an official Google product*
