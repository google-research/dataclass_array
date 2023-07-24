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

"""Dataclass array."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Callable, ClassVar, Generic, Iterator, Optional, Set, Tuple, Type, TypeVar, Union

from dataclass_array import field_utils
from dataclass_array import shape_parsing
from dataclass_array import type_parsing
from dataclass_array.typing import Array
from dataclass_array.typing import Axes, DTypeArg, DcOrArray, DcOrArrayT, DynamicShape, Shape  # pylint: disable=g-multiple-import,g-importing-member
from dataclass_array.utils import np_utils
from dataclass_array.utils import py_utils
import einops
from etils import array_types
from etils import edc
from etils import enp
from etils import epy
import numpy as np
import typing_extensions
from typing_extensions import Annotated, Literal, TypeAlias  # pylint: disable=g-multiple-import


lazy = enp.lazy

# TODO(pytype): Should use `dca.typing.DcT` but bound does not work across
# modules.
_DcT = TypeVar('_DcT', bound='DataclassArray')

# Any valid numpy indices slice ([x], [x:y], [:,...], ...)
_IndiceItem = Union[type(Ellipsis), None, int, slice, Any]
_Indices = Tuple[_IndiceItem]  # Normalized slicing
_IndicesArg = Union[_IndiceItem, _Indices]

_METADATA_KEY = 'dca_field'
_DUMMY_ARRAY_FIELD = '_dca_dummy_array'


@dataclasses.dataclass(frozen=True)
class DataclassParams:
  """Params controlling the DataclassArray behavior.

  Set by `@dca.dataclass_array`. Saved in `cls.__dca_params__`.

  Attributes:
    broadcast: If `True`, enable input broadcasting
    cast_dtype: If `True`, auto-cast inputs `dtype`
    cast_list: If `True`, auto-cast lists to `xnp.ndarray`
  """

  # If modifying this, make sure to modify `@dataclass_array` too!
  broadcast: bool = False
  cast_dtype: bool = False
  cast_list: bool = True


def dataclass_array(
    *,
    # If modifying this, make sure to modify `DataclassParams` too!
    broadcast: bool = False,
    cast_dtype: bool = False,
    cast_list: bool = True,
) -> Callable[[type[_DcT]], type[_DcT]]:
  """Optional decorator to customize `dca.DataclassArray` params.

  Usage:

  ```python
  @dca.dataclass_array()
  class MyDataclass(dca.DataclassArray):
    ...
  ```

  This decorator has to be added in addition of inheriting from
  `dca.DataclassArray`.

  Args:
    broadcast: If `True`, enable input broadcasting
    cast_dtype: If `True`, auto-cast inputs `dtype`
    cast_list: If `True`, auto-cast lists to `xnp.ndarray`

  Returns:
    decorator: The decorator which will apply the options to the dataclass
  """

  def decorator(cls):
    if not issubclass(cls, DataclassArray):
      raise TypeError(
          '`@dca.dataclass_array` can only be applied on `dca.DataclassArray`. '
          f'Got: {cls}'
      )
    cls.__dca_params__ = DataclassParams(
        broadcast=broadcast,
        cast_dtype=cast_dtype,
        cast_list=cast_list,
    )
    return cls

  return decorator


def array_field(
    shape: Shape,
    dtype: DTypeArg = float,
    **field_kwargs,
) -> dataclasses.Field[DcOrArray]:
  """Dataclass array field.

  See `dca.DataclassArray` for example.

  Args:
    shape: Inner shape of the field
    dtype: Type of the field
    **field_kwargs: Args forwarded to `dataclasses.field`

  Returns:
    The dataclass field.
  """
  # TODO(epot): Validate shape, dtype
  dca_field = _ArrayFieldMetadata(
      inner_shape_non_static=shape,
      dtype=dtype,
  )
  return dataclasses.field(**field_kwargs, metadata={_METADATA_KEY: dca_field})


class MetaDataclassArray(type):
  """DataclassArray metaclass."""

  # TODO(b/204422756): We cannot use `__class_getitem__` due to b/204422756
  def __getitem__(cls, spec):
    # Not clear how this would interact if cls is also a `Generic`
    return Annotated[cls, field_utils.ShapeAnnotation(spec)]


# TODO(epot): Restore once pytype support this
# @typing_extensions.dataclass_transform(
#     kw_only_default=True,
#     # TODO(b/272524683):Restore field specifier
#     # field_specifiers=(
#     #     dataclasses.Field,
#     #     dataclasses.field,
#     #     array_field,
#     # ),
# )
class DataclassArray(metaclass=MetaDataclassArray):
  """Dataclass which behaves like an array.

  Usage:

  ```python
  class Square(DataclassArray):
    pos: f32['*shape 2']
    scale: f32['*shape']
    name: str

  # Create 3 squares batched
  p = Square(
      pos=[[x0, y0], [x1, y1], [x2, y2]],
      scale=[scale0, scale1, scale2],
      name='my_square',
  )
  p.shape == (3,)
  p.pos.shape == (3, 2)
  p[0] == Square(pos=[x0, y0], scale=scale0)

  p = p.reshape((3, 1))  # Reshape the inner-shape
  p.shape == (3, 1)
  p.pos.shape == (3, 1, 2)

  p.name == 'my_square'
  ```

  `DataclassArray` has 2 types of fields:

  * Array fields: Fields batched like numpy arrays, with reshape, slicing,...
    (`pos` and `scale` in the above example).
  * Static fields: Other non-numpy field. Are not modified by reshaping,... (
    `name` in the above example).
    Static fields are also ignored in `jax.tree_map`.

  `DataclassArray` detect array fields if either:

  * The typing annotation is a `etils.array_types` annotation (in which
    case shape/dtype are automatically infered from the typing annotation)
    Example: `x: f32[..., 3]`
  * The typing annotation is another `dca.DataclassArray` (in which case
    `my_dataclass.field.shape == my_dataclass.shape`)
    Example: `x: MyDataclass`
  * The field is explicitly defined in `dca.array_field`, in which case
    the typing annotation is ignored.
    Example: `x: Any = dca.field(shape=(), dtype=np.int64)`

  Field which do not satisfy any of the above conditions are static (including
  field annotated with `field: np.ndarray` or similar).
  """

  # Child class inherit the default params by default, but can also
  # overwrite them.
  __dca_params__: ClassVar[DataclassParams] = DataclassParams()

  # TODO(epot): Could be removed with py3.10 and using `kw_only=True`
  # Fields defined here will be forwarded with `.replace`
  # TODO(py39): Replace Set -> set
  __dca_non_init_fields__: ClassVar[Set[str]] = set()

  _shape: Shape
  _xnp: enp.NpModule

  def __init_subclass__(
      cls,
      frozen=True,
      **kwargs,
  ):
    super().__init_subclass__(**kwargs)

    if not frozen:
      raise ValueError(f'{cls} cannot be `frozen=False`.')

    # Apply dataclass (in-place)
    if not typing.TYPE_CHECKING:
      # TODO(b/227290126): Create pytype issues
      dataclasses.dataclass(frozen=True)(cls)

    # TODO(epot): Could have smart __repr__ which display types if array have
    # too many values (maybe directly in `edc.field(repr=...)`).
    edc.dataclass(kw_only=True, repr=True, auto_cast=False)(cls)
    cls._dca_jax_tree_registered = False
    cls._dca_torch_tree_registered = False
    # Typing annotations have to be lazily evaluated (to support
    # `from __future__ import annotations` and forward reference)
    # To avoid costly `typing.get_type_hints` which perform `eval` and `str`
    # convertions, we cache the type annotations here.
    cls._dca_fields_metadata: Optional[dict[str, _ArrayFieldMetadata]] = None

    # Normalize the `cls.__dca_non_init_fields__`
    # TODO(epot): Support inheritance if the parents also define
    # `__dca_non_init_fields__` (fields should be merged from `.mro()`)
    cls.__dca_non_init_fields__ = set(cls.__dca_non_init_fields__)

  if typing.TYPE_CHECKING:
    # TODO(b/242839979): pytype do not support PEP 681 -- Data Class Transforms
    def __init__(self, **kwargs):
      pass

  def __post_init__(self) -> None:
    """Validate and normalize inputs."""
    cls = type(self)

    # First time, we perform additional check & updates
    if cls._dca_fields_metadata is None:  # pylint: disable=protected-access
      _init_cls(self)

    # Register the tree_map here instead of `__init_subclass__` as `jax` may
    # not have been imported yet during import.
    if enp.lazy.has_jax and not cls._dca_jax_tree_registered:  # pylint: disable=protected-access
      enp.lazy.jax.tree_util.register_pytree_node_class(cls)
      cls._dca_jax_tree_registered = True  # pylint: disable=protected-access

    if enp.lazy.has_torch and not cls._dca_torch_tree_registered:  # pylint: disable=protected-access
      # Note: Torch is updating it's tree API to make it public and use `optree`
      # as backend: https://github.com/pytorch/pytorch/issues/65761
      enp.lazy.torch.utils._pytree._register_pytree_node(  # pylint: disable=protected-access
          cls,
          flatten_fn=lambda a: a.tree_flatten(),
          unflatten_fn=lambda vals, ctx: cls.tree_unflatten(ctx, vals),
      )
      cls._dca_torch_tree_registered = True  # pylint: disable=protected-access

    # Validate and normalize array fields
    # * Maybe cast (list, np) -> xnp
    # * Maybe cast dtype
    # * Maybe broadcast shapes
    # Because this is only done inside `__init__`, it is ok to mutate self.

    # Cast and validate the array xnp are consistent
    xnp = self._cast_xnp_dtype_inplace()

    # Validate the batch shape is consistent
    # However, we need to be careful that `_ArrayField` never uses
    # `@epy.cached_property`
    shape = self._broadcast_shape_inplace()

    # TODO(epot): When to validate (`field.validate()`)

    if xnp is None:  # No values
      # Inside `jax.tree_utils`, tree-def can be created with `None` values.
      # Inside `jax.vmap`, tree can be created with `object()` sentinel values.
      assert shape is None
      xnp = None

    # Cache results
    # Should the state be stored in a separate object to avoid collisions ?
    assert shape is None or isinstance(shape, tuple), shape
    self._setattr('_shape', shape)
    self._setattr('_xnp', xnp)

  # ====== Array functions ======

  @property
  def shape(self) -> Shape:
    """Returns the batch shape common to all fields."""
    return self._shape

  @property
  def size(self) -> int:
    """Returns the number of elements."""
    return np_utils.size_of(self._shape)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions."""
    return len(self._shape)

  def reshape(self: _DcT, shape: Union[Shape, str], **axes_length: int) -> _DcT:
    """Reshape the batch shape according to the pattern.

    Supports both tuple and einops mode:

    ```python
    rays.reshape('b h w -> b (h w)')
    rays.reshape((128, -1))
    ```

    Args:
      shape: Target shape. Can be string for `einops` support.
      **axes_length: Any additional specifications for dimensions for einops
        support.

    Returns:
      The dataclass array with the new shape
    """
    if isinstance(shape, str):  # Einops support
      return self._map_field(  # pylint: disable=protected-access
          array_fn=lambda f: einops.rearrange(  # pylint: disable=g-long-lambda
              f.value,
              np_utils.to_absolute_einops(shape, nlastdim=len(f.inner_shape)),
              **axes_length,
          ),
          dc_fn=lambda f: f.value.reshape(  # pylint: disable=g-long-lambda
              np_utils.to_absolute_einops(shape, nlastdim=len(f.inner_shape)),
              **axes_length,
          ),
      )
    else:  # Numpy support
      assert isinstance(shape, tuple)  # For pytest

      def _reshape(f: _ArrayField):
        return f.value.reshape(shape + f.inner_shape)

      return self._map_field(array_fn=_reshape, dc_fn=_reshape)  # pylint: disable=protected-access

  def flatten(self: _DcT) -> _DcT:
    """Flatten the batch shape."""
    return self.reshape((-1,))

  def broadcast_to(self: _DcT, shape: Shape) -> _DcT:
    """Broadcast the batch shape."""
    return self._map_field(  # pylint: disable=protected-access
        array_fn=lambda f: f.broadcast_to(shape),
        dc_fn=lambda f: f.broadcast_to(shape),
    )

  def __getitem__(self: _DcT, indices: _IndicesArg) -> _DcT:
    """Slice indexing."""
    indices = np.index_exp[indices]  # Normalize indices
    # Replace `...` by explicit shape
    indices = _to_absolute_indices(indices, shape=self.shape)
    return self._map_field(
        array_fn=lambda f: f.value[indices],
        dc_fn=lambda f: f.value[indices],
    )

  # _DcT[n *d] -> Iterator[_DcT[*d]]
  def __iter__(self: _DcT) -> Iterator[_DcT]:
    """Iterate over the outermost dimension."""
    if not self.shape:
      raise TypeError(f'iteration over 0-d array: {self!r}')

    # Similar to `etree.unzip(self)` (but work with any backend)
    field_names = [f.name for f in self._array_fields]  # pylint: disable=not-an-iterable
    field_values = [f.value for f in self._array_fields]  # pylint: disable=not-an-iterable
    for vals in zip(*field_values):
      yield self.replace(**dict(zip(field_names, vals)))

  def __len__(self) -> int:
    """Length of the first array dimension."""
    if not self.shape:
      raise TypeError(
          f'len() of unsized {self.__class__.__name__} (shape={self.shape})'
      )
    return self.shape[0]

  def __bool__(self) -> Literal[True]:
    """`dca.DataclassArray` always evaluate to `True`.

    Like all python objects (including dataclasses), `dca.DataclassArray` always
    evaluate to `True`. So:
    `Ray(pos=None)`, `Ray(pos=0)` all evaluate to `True`.

    This allow construct like:

    ```python
    def fn(ray: Optional[dca.Ray] = None):
      if ray:
        ...
    ```

    Or:

    ```python
    def fn(ray: Optional[dca.Ray] = None):
      ray = ray or default_ray
    ```

    Only in the very rare case of empty-tensor (`shape=(0, ...)`)

    ```python
    assert ray is not None
    assert len(ray) == 0
    bool(ray)  # TypeError: Truth value is ambigous
    ```

    Returns:
      True

    Raises:
      ValueError: If `len(self) == 0` to avoid ambiguity.
    """
    if self.shape and not len(self):  # pylint: disable=g-explicit-length-test
      raise ValueError(
          f'The truth value of {self.__class__.__name__} when `len(x) == 0` '
          'is ambigous. Use `len(x)` or `x is not None`.'
      )
    return True

  def map_field(
      self: _DcT,
      fn: Callable[[Array['*din']], Array['*dout']],
  ) -> _DcT:
    """Apply a transformation on all arrays from the fields."""
    return self._map_field(  # pylint: disable=protected-access
        array_fn=lambda f: fn(f.value),
        dc_fn=lambda f: f.value.map_field(fn),
    )

  # ====== Dataclass/Conversion utils ======

  def replace(self: _DcT, **kwargs: Any) -> _DcT:
    """Alias for `dataclasses.replace`."""
    init_kwargs = {
        k: v for k, v in kwargs.items() if k not in self.__dca_non_init_fields__
    }
    non_init_kwargs = {
        k: v for k, v in kwargs.items() if k in self.__dca_non_init_fields__
    }

    # Create the new object
    new_self = dataclasses.replace(self, **init_kwargs)  # pytype: disable=wrong-arg-types  # re-none

    # TODO(epot): Could try to unify logic bellow with `tree_unflatten`

    # Additionally forward the non-init kwargs
    # `dataclasses.field(init=False) kwargs are required because `init=True`
    # creates conflicts:
    # * Inheritance fails with non-default argument 'K' follows default argument
    # * Pytype complains too
    # TODO(py310): Cleanup using `dataclass(kw_only)`
    assert new_self is not self
    for k in self.__dca_non_init_fields__:
      if k in non_init_kwargs:
        v = non_init_kwargs[k]
      else:
        v = getattr(self, k)
      new_self._setattr(k, v)  # pylint: disable=protected-access
    return new_self

  def as_np(self: _DcT) -> _DcT:
    """Returns the instance as containing `np.ndarray`."""
    return self.as_xnp(enp.lazy.np)

  def as_jax(self: _DcT) -> _DcT:
    """Returns the instance as containing `jnp.ndarray`."""
    return self.as_xnp(enp.lazy.jnp)

  def as_tf(self: _DcT) -> _DcT:
    """Returns the instance as containing `tf.Tensor`."""
    return self.as_xnp(enp.lazy.tnp)

  def as_torch(self: _DcT) -> _DcT:
    """Returns the instance as containing `torch.Tensor`."""
    return self.as_xnp(enp.lazy.torch)

  def as_xnp(self: _DcT, xnp: enp.NpModule) -> _DcT:
    """Returns the instance as containing `xnp.ndarray`."""
    if xnp is self.xnp:  # No-op
      return self
    # Direct `torch` <> `tf`/`jax` conversion not supported, so convert to
    # `numpy`
    if enp.lazy.is_torch_xnp(xnp) or enp.lazy.is_torch_xnp(self.xnp):

      def _as_torch(f):
        arr = np.asarray(f.value)
        # Torch fail for scalar arrays:
        # https://github.com/pytorch/pytorch/issues/97021
        if enp.lazy.is_torch_xnp(xnp) and not arr.shape:  # Destination is torch
          return xnp.asarray(arr.item(), dtype=lazy.as_torch_dtype(arr.dtype))

        return xnp.asarray(arr)

      array_fn = _as_torch
    else:
      array_fn = lambda f: xnp.asarray(f.value)

    # Update all childs
    new_self = self._map_field(  # pylint: disable=protected-access
        array_fn=array_fn,
        dc_fn=lambda f: f.value.as_xnp(xnp),
    )
    return new_self

  # TODO(pytype): Remove hack. Currently, Python does not support typing
  # annotations for modules, by pytype auto-infer the correct type.
  # So this hack allow auto-completion

  if typing.TYPE_CHECKING:

    @property
    def xnp(self):  # pylint: disable=function-redefined
      """Returns the numpy module of the class (np, jnp, tnp)."""
      return np

  else:

    @property
    def xnp(self) -> enp.NpModule:
      """Returns the numpy module of the class (np, jnp, tnp)."""
      return self._xnp

  # ====== Torch specific methods ======
  # Could also add
  # * x.detach
  # * x.is_cuda
  # * x.device
  # * x.get_device

  def to(self: _DcT, device, **kwargs) -> _DcT:
    """Move the dataclass array to the device."""
    if not lazy.is_torch_xnp(self.xnp):
      raise ValueError('`.to` can only be called when `xnp == torch`')
    return self.map_field(lambda f: f.to(device, **kwargs))

  def cpu(self: _DcT, *args, **kwargs) -> _DcT:
    """Move the dataclass array to the CPU device."""
    if not lazy.is_torch_xnp(self.xnp):
      raise ValueError('`.cpu` can only be called when `xnp == torch`')
    return self.map_field(lambda f: f.cpu(*args, **kwargs))

  def cuda(self: _DcT, *args, **kwargs) -> _DcT:
    """Move the dataclass array to the CUDA device."""
    if not lazy.is_torch_xnp(self.xnp):
      raise ValueError('`.cuda` can only be called when `xnp == torch`')
    return self.map_field(lambda f: f.cuda(*args, **kwargs))

  # ====== Internal ======

  @epy.cached_property
  def _all_fields_empty(self) -> bool:
    """Returns True if the `dataclass_array` is invalid."""
    if not self._array_fields:  # All fields are `None` / `object`
      # No fields have been defined.
      # This can be the case internally by jax which apply some
      # `tree_map(lambda x: sentinel)`.
      return True

    # `tf.nest` sometimes replace values by dummy `.` inside
    # `assert_same_structure`
    if enp.lazy.has_tf:
      # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
      from tensorflow.python.util import nest_util  # pytype: disable=import-error
      # pylint: enable=g-direct-tensorflow-import,g-import-not-at-top

      if any(f.value is nest_util._DOT for f in self._array_fields):  # pylint: disable=protected-access,not-an-iterable
        return True
    return False

  @epy.cached_property
  def _all_array_fields(self) -> dict[str, _ArrayField]:
    """All array fields, including `None` values."""
    return {  # pylint: disable=g-complex-comprehension
        name: _ArrayField(
            name=name,
            host=self,
            **field_metadata.to_dict(),  # pylint: disable=not-a-mapping
        )
        for name, field_metadata in self._dca_fields_metadata.items()  # pylint: disable=protected-access
    }

  @epy.cached_property
  def _array_fields(self) -> list[_ArrayField]:
    """All active array fields (non-None), including static ones."""
    # Filter `None` values
    return [
        f for f in self._all_array_fields.values() if not f.is_value_missing
    ]

  def _cast_xnp_dtype_inplace(self) -> Optional[enp.NpModule]:
    """Validate `xnp` are consistent and cast `np` -> `xnp` in-place."""
    if self._all_fields_empty:  # pylint: disable=using-constant-test
      return None

    # Validate the dtype
    def _get_xnp(f: _ArrayField) -> enp.NpModule:
      try:
        return np_utils.get_xnp(
            f.value,
            strict=not self.__dca_params__.cast_list,
        )
      except Exception as e:  # pylint: disable=broad-except
        epy.reraise(e, prefix=f'Invalid {f.qualname}: ')

    xnps = epy.groupby(
        self._array_fields,
        key=_get_xnp,
        value=lambda f: f.name,
    )
    if not xnps:
      return None
    xnp = _infer_xnp(xnps)

    def _cast_field(f: _ArrayField) -> None:
      try:
        # Supports for TensorSpec (e.g. in `tf.function` signature)
        if enp.lazy.is_tf_xnp(xnp) and isinstance(
            f.value, enp.lazy.tf.TensorSpec
        ):
          # TODO(epot): Actually check the dtype
          new_value = f.value
        else:
          new_value = np_utils.asarray(
              f.value,
              xnp=xnp,
              dtype=f.dtype,
              cast_dtype=self.__dca_params__.cast_dtype,
          )
        self._setattr(f.name, new_value)
        # After the field has been set, we validate the shape
        f.assert_shape()
      except Exception as e:  # pylint: disable=broad-except
        epy.reraise(e, prefix=f'Invalid {f.qualname}: ')

    self._map_field(
        array_fn=_cast_field,
        dc_fn=_cast_field,  # pytype: disable=wrong-arg-types
        _inplace=True,
    )
    return xnp

  def _broadcast_shape_inplace(self) -> Optional[Shape]:
    """Validate the shapes are consistent and broadcast values in-place."""
    if self._all_fields_empty:  # pylint: disable=using-constant-test
      return None

    # First collect all shapes and compute the final shape.
    shape_to_names = epy.groupby(
        self._array_fields,
        key=lambda f: f.host_shape,
        value=lambda f: f.name,
    )
    shape_lengths = {len(s) for s in shape_to_names.keys()}

    # Broadcast all shape together
    try:
      final_shape = np.broadcast_shapes(*shape_to_names.keys())
    except ValueError:
      final_shape = None  # Bad broadcast

    # Currently, we restrict broadcasting to either scalar or fixed length.
    # This is to avoid confusion broadcasting vs vectorization rules.
    # This restriction could be lifted if we encounter a use-case.
    if (
        final_shape is None
        or len(shape_lengths) > 2
        or (len(shape_lengths) == 2 and 0 not in shape_lengths)
    ):
      raise ValueError(
          f'Conflicting batch shapes: {shape_to_names}. '
          f'Currently {type(self).__qualname__}.__init__ broadcasting is '
          'restricted to scalar or dim=1 . '
          'Please open an issue if you need more fine-grained broadcasting.'
      )

    def _broadcast_field(f: _ArrayField) -> None:
      if f.host_shape == final_shape:  # Already broadcasted
        return
      elif not self.__dca_params__.broadcast:  # Broadcasing disabled
        raise ValueError(
            f'{type(self).__qualname__} has `broadcast=False`. '
            f'Cannot broadcast {f.name} from {f.full_shape} to {final_shape}. '
            'To enable broadcast, use `@dca.dataclass_array(broadcast=True)`.'
        )
      self._setattr(f.name, f.broadcast_to(final_shape))

    self._map_field(
        array_fn=_broadcast_field,
        dc_fn=_broadcast_field,  # pytype: disable=wrong-arg-types
        _inplace=True,
    )
    return final_shape

  def _to_absolute_axis(self, axis: Axes) -> Axes:
    """Normalize the axis to absolute value."""
    try:
      return np_utils.to_absolute_axis(axis, ndim=self.ndim)
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(
          e,
          prefix=f'For {self.__class__.__qualname__} with shape={self.shape}: ',
      )

  def _map_field(
      self: _DcT,
      *,
      array_fn: Callable[[_ArrayField[Array['*din']]], Array['*dout']],
      dc_fn: Optional[Callable[[_ArrayField[_DcT]], _DcT]],
      _inplace: bool = False,
  ) -> _DcT:
    """Apply a transformation on all array fields structure.

    Args:
      array_fn: Function applied on the `xnp.ndarray` fields
      dc_fn: Function applied on the `dca.DataclassArray` fields (to recurse)
      _inplace: If True, assume the function mutate the object in-place. Should
        only be used inside `__init__` for performances.

    Returns:
      The transformed dataclass array.
    """

    def _apply_field_dn(f: _ArrayField):
      if f.is_dataclass:  # Recurse on dataclasses
        return dc_fn(f)  # pylint: disable=protected-access
      else:
        return array_fn(f)

    new_values = {f.name: _apply_field_dn(f) for f in self._array_fields}  # pylint: disable=not-an-iterable,protected-access
    # For performance, do not call replace to save the constructor call
    if not _inplace:
      return self.replace(**new_values)
    else:
      return self

  def tree_flatten(self) -> tuple[tuple[DcOrArray, ...], _TreeMetadata]:
    """`jax.tree_utils` support."""
    # We flatten all values (and not just the non-None ones)
    array_field_values = tuple(f.value for f in self._all_array_fields.values())
    metadata = _TreeMetadata(
        array_field_names=list(self._all_array_fields.keys()),
        non_array_field_kwargs={
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)  # pytype: disable=wrong-arg-types  # re-none
            if f.name not in self._all_array_fields  # pylint: disable=unsupported-membership-test
        },
    )
    return (array_field_values, metadata)

  @classmethod
  def tree_unflatten(
      cls: Type[_DcT],
      metadata: _TreeMetadata,
      array_field_values: list[DcOrArray],
  ) -> _DcT:
    """`jax.tree_utils` support."""
    array_field_kwargs = dict(
        zip(
            metadata.array_field_names,
            array_field_values,
        )
    )
    init_fields = {}
    non_init_fields = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}  # pytype: disable=wrong-arg-types  # re-none
    for k, v in metadata.non_array_field_kwargs.items():
      if fields[k].init:
        init_fields[k] = v
      else:
        non_init_fields[k] = v

    self = cls(**array_field_kwargs, **init_fields)
    # Currently it's not clear how to handle non-init fields so raise an error
    if non_init_fields:
      if set(non_init_fields) - self.__dca_non_init_fields__:
        raise ValueError(
            '`dca.DataclassArray` field with init=False should be explicitly '
            'specified in `__dca_non_init_fields__` for them to be '
            'propagated by `tree_map`.'
        )
      # TODO(py310): Delete once dataclass supports `kw_only=True`
      for k, v in non_init_fields.items():
        self._setattr(k, v)  # pylint: disable=protected-access
    return self

  def __tf_flatten__(self) -> tuple[_TreeMetadata, tuple[DcOrArray, ...]]:
    components, metadata = self.tree_flatten()
    return metadata, components

  @classmethod
  def __tf_unflatten__(
      cls: Type[_DcT],
      metadata: _TreeMetadata,
      components: list[DcOrArray],
  ) -> _DcT:
    return cls.tree_unflatten(metadata, components)

  def _setattr(self, name: str, value: Any) -> None:
    """Like setattr, but support `frozen` dataclasses."""
    object.__setattr__(self, name, value)

  def assert_same_xnp(self, x: Union[Array[...], DataclassArray]) -> None:
    """Assert the given array is of the same type as the current object."""
    xnp = np_utils.get_xnp(x)
    if xnp is not self.xnp:
      raise ValueError(
          f'{self.__class__.__name__} is {self.xnp.__name__} but got input '
          f'{xnp.__name__}. Please cast input first.'
      )


def _init_cls(self: DataclassArray) -> None:
  """Setup the class the first time the instance is called.

  This will:

  * Extract the types annotations, detect which fields are arrays or static,
    and store the result in `_dca_fields_metadata`
  * For static `DataclassArray` (class with only static fields), it will
    add a dummy array field for compatibility with `.xnp`/`.shape` (so
    methods works correctly and return the right shape/xnp when nested)

  Args:
    self: The dataclass to initialize
  """
  cls = type(self)

  # The first time, compute typing annotations & metadata
  # At this point, `ForwardRef` should have been resolved.
  try:
    hints = typing_extensions.get_type_hints(cls, include_extras=True)
  except Exception as e:  # pylint: disable=broad-except
    msg = (
        f'Could not infer typing annotation of {cls.__qualname__} '
        f'defined in {cls.__module__}:\n'
    )
    lines = [f' * {k}: {v!r}' for k, v in cls.__annotations__.items()]
    lines = '\n'.join(lines)

    epy.reraise(e, prefix=msg + lines + '\n')

  # TODO(epot): Remove restriction once pytype supports `datclass_transform`
  # and `dca` automatically apply the `@dataclasses.dataclass`
  if _DUMMY_ARRAY_FIELD in cls.__dataclass_fields__:  # pytype: disable=attribute-error
    raise NotImplementedError(
        'Suclassing of DataclassArray with no array field is not supported '
        'after an instance of the class was created. Error raised for '
        f'{cls.__qualname__}'
    )

  dca_fields_metadata = {
      f.name: _make_field_metadata(f, hints) for f in dataclasses.fields(cls)  # pytype: disable=wrong-arg-types
  }
  dca_fields_metadata = {  # Filter `None` values (static fields)
      k: v for k, v in dca_fields_metadata.items() if v is not None
  }
  if not dca_fields_metadata:
    # DataclassArray without any array fields
    # Hack: To support `.xnp`, `.shape`, we add a dummy empty field which
    # is propagated by the various ops.
    dca_fields_metadata[_DUMMY_ARRAY_FIELD] = _ArrayFieldMetadata(  # pytype: disable=wrong-arg-types
        inner_shape_non_static=(),
        dtype=np.float32,
    )
    default_dummy_array = np.zeros((), dtype=np.float32)
    _add_field_to_dataclass(
        cls, _DUMMY_ARRAY_FIELD, default=default_dummy_array
    )
    # Because we're in `__init__`, so also update the current call
    self._setattr(_DUMMY_ARRAY_FIELD, default_dummy_array)  # pylint: disable=protected-access

  cls._dca_fields_metadata = (  # pylint: disable=protected-access
      dca_fields_metadata
  )


def _add_field_to_dataclass(cls, name: str, default: Any) -> None:
  """Add a new field to the given dataclass."""
  # Make sure to not update the parent class
  # Otherwise we could even accidentally update `dca.DataclassArray`
  if '__dataclass_fields__' not in cls.__dict__:
    # TODO(epot): Remove the limitation once `dataclasses.dataclass` is
    # automatically applied
    raise ValueError(
        f'{cls.__name__} is not a `@dataclasses.dataclass(frozen=True)`'
    )
  assert name not in cls.__dataclass_fields__  # pytype: disable=attribute-error

  # Ideally, we want init=False, so sub-dataclass ignore this field
  # but this makes `.replace` fail
  field = dataclasses.field(default=default, init=True, repr=False)
  field.__set_name__(cls, name)
  field.name = name
  field.type = Any
  field._field_type = dataclasses._FIELD  # pylint: disable=protected-access  # pytype: disable=module-attr
  cls.__dataclass_fields__[name] = field  # pytype: disable=attribute-error

  original_init = cls.__init__

  @functools.wraps(original_init)
  def new_init(self, **kwargs: Any):
    self._setattr(name, kwargs.pop(name, default))  # pylint: disable=protected-access
    return original_init(self, **kwargs)

  cls.__init__ = new_init


def _infer_xnp(xnps: dict[enp.NpModule, list[str]]) -> enp.NpModule:
  """Extract the `xnp` module."""
  non_np_xnps = set(xnps) - {np}  # jnp, tnp take precedence on `np`

  # Detecting conflicting xnp
  if len(non_np_xnps) > 1:
    xnps = {k.__name__: v for k, v in xnps.items()}
    raise ValueError(f'Conflicting numpy types: {xnps}')

  if not non_np_xnps:
    return np
  else:
    (xnp,) = non_np_xnps
    return xnp


def _count_not_none(indices: _Indices) -> int:
  """Count the number of non-None and non-ellipsis elements."""
  return len([k for k in indices if k is not np.newaxis and k is not Ellipsis])


def _count_ellipsis(elems: _Indices) -> int:
  """Returns the number of `...` in the indices."""
  # Cannot use `elems.count(Ellipsis)` because `np.array() == Ellipsis` fail
  return len([elem for elem in elems if elem is Ellipsis])


def _to_absolute_indices(indices: _Indices, *, shape: Shape) -> _Indices:
  """Normalize the indices to replace `...`, by `:, :, :`."""
  assert isinstance(indices, tuple)
  ellipsis_count = _count_ellipsis(indices)
  if ellipsis_count > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  valid_count = _count_not_none(indices)
  if valid_count > len(shape):
    raise IndexError(
        f'too many indices for array. Batch shape is {shape}, but '
        f'rank-{valid_count} was provided.'
    )
  if not ellipsis_count:
    return indices
  ellipsis_index = indices.index(Ellipsis)
  start_elems = indices[:ellipsis_index]
  end_elems = indices[ellipsis_index + 1 :]
  ellipsis_replacement = [slice(None)] * (len(shape) - valid_count)
  return (*start_elems, *ellipsis_replacement, *end_elems)


@dataclasses.dataclass(frozen=True)
class _TreeMetadata:
  """Metadata forwarded in ``."""

  array_field_names: list[str]
  non_array_field_kwargs: dict[str, Any]


# TODO(epot): Should refactor `_ArrayField` in `_DataclassArrayField` and
# `_ArrayField` depending on whether dtype is `DataclassArray` or not.
# Alternativelly, maybe should create a `DcArrayDType` dtype instead.


@edc.dataclass
@dataclasses.dataclass
class _ArrayFieldMetadata:
  """Metadata of the array field (shared across all instances).

  Attributes:
    inner_shape_non_static: Inner shape. Can contain non-static dims (e.g.
      `(None, 3)`)
    dtype: Type of the array. Can be `enp.dtypes.DType` or
      `dca.DataclassArray` for nested arrays.
  """

  inner_shape_non_static: DynamicShape
  dtype: Union[enp.dtypes.DType, Type[DataclassArray]]

  def __post_init__(self):
    """Normalizing/validating the shape/dtype."""
    # Validate shape
    self.inner_shape_non_static = tuple(self.inner_shape_non_static)

    # Validate/normalize the dtype
    if not self.is_dataclass:
      self.dtype = enp.dtypes.DType.from_value(self.dtype)
      # TODO(epot): Filter invalid dtypes, like `str` ?

  def to_dict(self) -> dict[str, Any]:
    """Returns the dict[field_name, field_value]."""
    return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

  @property
  def is_dataclass(self) -> bool:
    """Returns `True` if the field is a dataclass."""
    return epy.issubclass(self.dtype, DataclassArray)


@edc.dataclass
@dataclasses.dataclass
class _ArrayField(_ArrayFieldMetadata, Generic[DcOrArrayT]):
  """Array field of a specific dataclass instance.

  Attributes:
    name: Instance of the attribute
    host: Dataclass instance who this field is attached too
  """

  name: str
  host: DataclassArray = dataclasses.field(repr=False)

  @property
  def qualname(self) -> str:
    """Returns the `'MyClass.attr_name'`."""
    return f'{type(self.host).__name__}.{self.name}'

  @property
  def xnp(self) -> enp.NpModule:
    """Numpy module of the field."""
    return np_utils.get_xnp(self.value)

  @property
  def value(self) -> DcOrArrayT:
    """Access the `host.<field-name>`."""
    return getattr(self.host, self.name)

  @property
  def full_shape(self) -> DcOrArrayT:
    """Access the `host.<field-name>.shape`."""
    # TODO(b/198633198): We need to convert to tuple because TF evaluate
    # empty shapes to True `bool(shape) == True` when `shape=()`
    return tuple(self.value.shape)

  @epy.cached_property
  def inner_shape(self) -> Shape:
    """Returns the the static shape resolved for the current value."""
    if not self.inner_shape_non_static:
      return ()
    static_shape = self.full_shape[-len(self.inner_shape_non_static) :]

    def err_msg() -> ValueError:
      return ValueError(
          f'Shape do not match. Expected: {self.inner_shape_non_static}. '
          f'Got {static_shape}'
      )

    if len(static_shape) != len(self.inner_shape_non_static):
      raise err_msg()
    for static_dim, non_static_dim in zip(  # Validate all dims
        static_shape,
        self.inner_shape_non_static,
    ):
      if non_static_dim is not None and non_static_dim != static_dim:
        raise err_msg()
    return static_shape

  @property
  def is_value_missing(self) -> bool:
    """Returns `True` if the value wasn't set."""
    if self.value is None:
      return True
    elif type(self.value) is object:  # pylint: disable=unidiomatic-typecheck
      # Checking for `object` is a hack required for `@jax.vmap` compatibility:
      # In `jax/_src/api_util.py` for `flatten_axes`, jax set all values to a
      # dummy sentinel `object()` value.
      return True
    elif (
        isinstance(self.value, DataclassArray) and not self.value._array_fields  # pylint: disable=protected-access
    ):
      # Nested dataclass case (if all attributes are `object`, so no active
      # array fields)
      return True
    return False

  @property
  def host_shape(self) -> Shape:
    """Host shape (batch shape shared by all fields)."""
    if not self.inner_shape_non_static:
      shape = self.full_shape
    else:
      shape = self.full_shape[: -len(self.inner_shape_non_static)]
    return shape

  def assert_shape(self) -> None:
    if self.host_shape + self.inner_shape != self.full_shape:
      raise ValueError(
          'Shape should be '
          f'{(py_utils.Ellipsis, *self.inner_shape)}. Got: {self.full_shape}'
      )

  def broadcast_to(self, shape: Shape) -> DcOrArrayT:
    """Broadcast the host_shape."""
    final_shape = shape + self.inner_shape
    if self.is_dataclass:
      return self.value.broadcast_to(final_shape)
    else:
      return self.xnp.broadcast_to(self.value, final_shape)


def _make_field_metadata(
    field: dataclasses.Field[Any],
    hints: dict[str, TypeAlias],
) -> Optional[_ArrayFieldMetadata]:
  """Make the array field class."""
  # TODO(epot): One possible confusion is if user define
  # `field: Ray = dca.array_field(shape=(3,))`
  # In which case field will be `float32` instead of `Ray`. Should we raise
  # a warning / error ?
  if _METADATA_KEY in field.metadata:  # Field defined as `= dca.array_field`:
    field_metadata = field.metadata[_METADATA_KEY]
  # TODO(py38):
  # elif field_metadata := _type_to_field_metadata(hints[field.name]):
  else:
    field_metadata = _type_to_field_metadata(hints[field.name])
    if not field_metadata:  # Not an array field
      return None

  return field_metadata


def _type_to_field_metadata(hint: TypeAlias) -> Optional[_ArrayFieldMetadata]:
  """Converts type hint to extract `inner_shape`, `dtype`."""
  array_type = type_parsing.get_array_type(hint)

  if isinstance(array_type, field_utils.DataclassWithShape):
    dtype = array_type.cls
  elif isinstance(array_type, array_types.ArrayAliasMeta):
    dtype = array_type.dtype
  else:  # Not a supported type: Static field
    return None

  try:
    return _ArrayFieldMetadata(
        inner_shape_non_static=shape_parsing.get_inner_shape(array_type.shape),
        dtype=dtype,
    )
  except Exception as e:  # pylint: disable=broad-except
    epy.reraise(e, prefix=f'Invalid shape annotation {hint}.')
