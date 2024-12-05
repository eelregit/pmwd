from abc import ABC, abstractmethod
from collections.abc import Callable
import dataclasses
from enum import Flag, auto
from functools import partial
from pprint import pformat

import jax.numpy as jnp
from jax import lax
from jax.tree_util import GetAttrKey, register_pytree_with_keys, tree_leaves, tree_map

from pmwd.util import add, sub, neg, scalar_mul, scalar_div


# FIXME DataDescriptor -> Data
#       DataTree -> Tree
#       any problem?


# FIXME tutorial idea:
#   linear regression to noisy data: z = 0 + x + eps
#       1. y = a + b x  # dyn_field
#       2. y = 0 + b x  # fxd_field
#       3. dtype from f8 to f4?  # aux_field
# show that fxd_field does not trigger re-compilation



# FIXME where to move validators and transformers? util.py? tree.py? transform.py and validate.py?


# TODO add serialization to zarr https://docs.xarray.dev/en/latest/user-guide/io.html#zarr
# TODO https://docs.python.org/3/library/json.html extend encoder and decoder, as zarr uses json
# TODO tensorstore is probably better


def issubdtype_of(dtype):
    """Return a function that raises `ValueError` if input object is not equal or lower
    than specified `dtype`.

    Useful for validation in `DataDescriptor`.

    """
    def fun(value):
        if not jnp.issubdtype(value, dtype):
            raise ValueError(f'{obj!r} must be sub-dtype of {dtype!r}')
        return value
    return fun


def asarray_of(dtype=None, field=None):
    """Return a function that converts input pytree children to JAX arrays of specified
    `dtype`, or dtype given by the specified `field` of pytree dataclass.

    Useful for validation in `DataDescriptor`. `dtype` can be `None` or `float`, while
    more specific conversions can be done using e.g. `jnp.float32` instead of
    ``asarray_of(dtype=jnp.float32)``.

    """
    if field is None:
        def fun(value):
            return tree_map(partial(jnp.asarray, dtype=dtype), value)
        return fun

    if dtype is not None:
        raise ValueError('dtype and field are mutually exclusive')

    def fun(value, obj):
        # FIXME are these line necessary/useful? this uses treemap and jnp anyway?
        #if not hasattr(obj, 'iter_fields'):
        #    raise TypeError(f'{obj=} not a pytree dataclass')
        #if field not in obj.iter_fields(ftype=FType.AUXILIARY, name=True):
        #    raise ValueError(f'{field=} not in auxiliary fields of {obj=}')
        dtype = getattr(obj, field)
        return tree_map(partial(jnp.asarray, dtype=dtype), value)
    return fun


def reshape_to(shape=None, field=None):
    """Return a function that reshapes input pytree children to specified `shape`, or
    shape given by the specified `field` of pytree dataclass.

    Useful for validation in `DataDescriptor`.

    """
    if field is None:
        def fun(value):
            return tree_map(partial(jnp.reshape, shape=shape), value)
        return fun

    if shape is not None:
        raise ValueError('shape and field are mutually exclusive')

    def fun(value, obj):
        # FIXME are these line necessary/useful? this uses treemap and jnp anyway?
        #if not hasattr(obj, iter_fields):
        #    raise TypeError(f'{obj=} not a pytree dataclass')
        #if field not in obj.iter_fields(ftype=FType.AUXILIARY, name=True):
        #    raise ValueError(f'{field=} not in auxiliary fields of {obj=}')
        shape = getattr(obj, field)
        return tree_map(partial(jnp.reshape, shape=shape), value)
    return fun


# TODO get inspirations from attrs, cattrs, pydantic, traitlets, marshmallow, schematics


def _canonicalize_callables(fun):
    """Canonicalize callables to a tuple of them."""
    if fun is None:
        fun = ()
    elif isinstance(fun, Callable):
        fun = (fun,)
    else:
        fun = tuple(fun)
    return fun


def _call_dual_arity(fun, value, obj):
    """Call as a binary function first and then as a unary function.

    ``fun(value, obj)``, ``fun(value)``, or fail.

    """
    try:
        return fun(value, obj)
    except TypeError as err:
        err_binary = err

    try:
        return fun(value)
    except TypeError as err:
        err_unary = err

    err = TypeError(f'calling {fun.__qualname__} fails on both binary and unary forms '
                    f'for {value=!r} and {obj=!r}')
    err.add_note(f'    binary: {err_binary}')
    err.add_note(f'    unary:  {err_unary!r}')
    raise err


class _BREAK_TYPE:
    """Sentinel type to signal breaking out of validation or transformation loops."""
BREAK = _BREAK_TYPE()


class DataDescriptor:
    """Data descriptor with computation rules.

    Parameters
    ----------
    mandatory : bool, optional
        Whether the value must be initialized to something other than `None`, if none of
        `default`, `default_function`, or `cache` is specified.
    default : pytree, optional
        Default value.
    default_function : callable or None, optional
        Default function to compute the value, if not already initialized to anything
        else but `None`, from the instance of the owner class, ``value =
        default_function(obj)``, runned by e.g. `DataTree.__post_init__`.
    cache : callable or None, optional
        Caching function to compute the value from the instance of the owner class,
        ``value = cache(obj)``, runned by e.g. `DataTree.cache`.
    validate : callable, sequence of callable, or None, optional
        Validator functions before setting the value, ``value = fun(value)`` or ``value
        = fun(value, obj)``. Skipped if input ``value is None``. If a sequence, apply
        each in turn. If any returns `BREAK`, break out of the loop.
    transform : callable, sequence of callable, or None, optional
        Transformer functions after getting the value, ``value = fun(value)`` or ``value
        = fun(value, obj)``. Skipped if input ``value is None``. If a sequence, apply
        each in turn. If any returns `BREAK`, break out of the loop. Useful with
        `jax.lax.stop_gradient`.

    Raises
    ------
    ValueError
        If more than one is specified among `default`, `default_function`, and `cache`,
        or if mandatory data is missing when none of the three is specified.
    TypeError
        If trying to set or delete descriptor attributes.

    References
    ----------
    .. _Python Descriptor HowTo Guide:
        https://docs.python.org/3/howto/descriptor.html

    .. _Python Data Model - Implementing Descriptors:
        https://docs.python.org/3/reference/datamodel.html#implementing-descriptors

    .. _Python Data Model - Invoking Descriptors:
        https://docs.python.org/3/reference/datamodel.html#invoking-descriptors

    .. _Python Data Classes - Descriptor-typed fields:
        https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields

    .. _Combining a descriptor class with dataclass and field:
        https://stackoverflow.com/questions/67612451/combining-a-descriptor-class-with-dataclass-and-field

    .. _Google etils dataclass field descriptor:
        https://github.com/google/etils/blob/main/etils/edc/field_utils.py

    .. _Dataclass descriptor behavior inconsistent:
        https://github.com/python/cpython/issues/102646

    """

    __slots__ = (
        'mandatory',
        'default',
        'default_function',
        'cache',
        'validate',
        'transform',
        'objtype',
        'name',
        '_name',
    )

    def __init__(self, mandatory=True, default=None, default_function=None, cache=None,
                 validate=None, transform=None):
        if sum(x is not None for x in (default, default_function, cache)) > 1:
            raise ValueError(f'{default=}, {default_function=}, and {cache=} are '
                             'mutually exclusive')

        validate = _canonicalize_callables(validate)
        transform = _canonicalize_callables(transform)

        object.__setattr__(self, 'mandatory', mandatory)
        object.__setattr__(self, 'default', default)
        object.__setattr__(self, 'default_function', default_function)
        object.__setattr__(self, 'cache', cache)
        object.__setattr__(self, 'validate', validate)
        object.__setattr__(self, 'transform', transform)

    def __repr__(self):
        return (
            f'{type(self).__qualname__}(\n'
            f'    mandatory={self.mandatory!r},\n'
            f'    default={self.default!r},\n'
            f'    default_function={self.default_function!r},\n'
            f'    cache={self.cache!r},\n'
            f'    validate={pformat(repr(self.validate))},\n'
            f'    transform={pformat(repr(self.transform))},\n'
            ')'
        )

    def __str__(self):
        return (f'<{self.objtype.__qualname__}.{self.name} decribed by ' + repr(self)
                + '>')

    def __setattr__(self, name, value):
        raise TypeError(f'{type(self)} is read-only')

    def __delattr__(self, name):
        raise TypeError(f'{type(self)} is read-only')

    def __set_name__(self, objtype, name):
        object.__setattr__(self, 'objtype', objtype)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, '_name', '_' + name)

    def __get__(self, obj, objtype=None):
        #print(f'__get__: {type(obj)=!r}, {obj=!r}', flush=True)  #FIXME convert to logging if needed
        #print(f'__get__: {type(objtype)=!r}, {objtype=!r}', flush=True)
        # DataDescriptor directly as field default value
        if obj is None:
            #print(f'__get__: obj is None, {objtype=!r}', flush=True)
            return self.default
        value = getattr(obj, self._name)
        return self.run_transform(value, obj)

    def __set__(self, obj, value):
        #try:
        #    print(f'__set__: {type(obj)=!r}, {obj=!r}', flush=True)
        #except AttributeError as err:
        #    print(f'__set__: {type(obj)=!r}, AttributeError: {err} when print(obj)', flush=True)
        #print(f'__set__: {type(value)=!r}, {value=!r}', flush=True)
        # field(default=DataDescriptor(...), init=True, ...)
        # "broken behavior" in cpython issue #102646
        if value is self:
            #print(f'__set__: {value=!r} is self', flush=True)
            value = self.default
        value = self.run_validate(value, obj)
        object.__setattr__(obj, self._name, value)

    def raise_missing(self, obj):
        """Raise if mandatory but missing."""
        if self.mandatory and all(x is None for x in (
                self.default, self.default_function, self.cache, self.__get__(obj))):
            raise ValueError(f'mandatory {self.name} missing for '
                             f'{self.objtype.__qualname__}')

    def run_default_function(self, obj):
        """Run default function."""
        if self.default_function is None or self.__get__(obj) is not None:
            return obj
        value = self.default_function(obj)
        return value

    def run_cache(self, obj):
        """Run caching function."""
        if self.cache is None:
            return obj
        value = self.cache(obj)
        return value

    def run_validate(self, value, obj):
        """Run validator functions."""
        if value is None:
            return value
        for validate in self.validate:
            value_ = _call_dual_arity(validate, value, obj)
            if value_ is BREAK:
                return value
            value = value_
        return value

    def run_transform(self, value, obj):
        """Run transformer functions."""
        if value is None:
            return value
        for transform in self.transform:
            value_ = _call_dual_arity(transform, value, obj)
            if value_ is BREAK:
                return value
            value = value_
        return value


def field(*, mandatory=True, default=None, default_function=None, cache=None,
             validate=None, transform=None, **kwargs):
    """Descriptor dataclass field.

    See `DataDescriptor` and `dataclasses.field` documentations. For JAX pytrees, use
    `dyn_field`, `fxd_field`, and `aux_field` instead.

    Parameters
    ----------
    **kwargs
        Parameters for `dataclasses.field` (with the other ones before them for
        `DataDescriptor`).

    """
    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        **kwargs,
    )


def break_on_jax_placeholder(obj):
    """Signal loop breaking on JAX transformation placeholders.

    References
    ----------
    .. _Pytrees — JAX documentation:
        https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

    .. _JAX Issue #10238:
        https://github.com/google/jax/issues/10238

    """
    return BREAK if type(obj) is object else obj


class FType(Flag):
    """Pytree dataclass field types."""
    DYNAMIC = auto()
    FIXED = auto()
    AUXILIARY = auto()
    CHILD = DYNAMIC | FIXED
FType.DYNAMIC.__doc__ = 'Dynamic pytree children with gradients.'
FType.FIXED.__doc__ = 'Fixed pytree children without gradients.'
FType.CHILD.__doc__ = 'Pytree children, dynamic or fixed.'
FType.AUXILIARY.__doc__ = 'Pytree auxiliary data stored in pytree treedef.'


def _update_metadata(kwargs, ftype):
    metadata = kwargs.pop('metadata', {})
    metadata = {} if metadata is None else dict(metadata)
    metadata['ftype'] = ftype
    kwargs['metadata'] = metadata
    return kwargs


def dyn_field(*, mandatory=True, default=None, default_function=None, cache=None,
              validate=None, transform=None, **kwargs):
    """Descriptor dataclass field for dynamic pytree children.

    `break_on_jax_placeholder` is prepended to `validate` and `transform` to skip on JAX
    transformation placeholders whose `type` is `object`. `dataclasses.Field.metadata`
    is updated with ``'ftype'``. See `DataDescriptor`, `dataclasses.field`, and JAX
    pytree documentations.

    Parameters
    ----------
    **kwargs
        Parameters for `dataclasses.field` (with the other ones before them for
        `DataDescriptor`).

    """
    validate = _canonicalize_callables(validate)
    transform = _canonicalize_callables(transform)

    validate = (break_on_jax_placeholder,) + validate
    transform = (break_on_jax_placeholder,) + transform

    kwargs = _update_metadata(kwargs, FType.DYNAMIC)

    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        **kwargs,
    )


def fxd_field(*, mandatory=True, default=None, default_function=None, cache=None,
              validate=None, transform=lax.stop_gradient, repr=False, **kwargs):
    """Descriptor dataclass field for fixed pytree children.

    `break_on_jax_placeholder` is prepended to `validate` and `transform` to skip on JAX
    transformation placeholders whose `type` is `object`. `lax.stop_gradient` is
    appended to `transform` if not already in it. `dataclasses.Field.metadata` is
    updated with ``'ftype'``. `repr` is supppressed by default. See `DataDescriptor`,
    `dataclasses.field`, and JAX pytree documentations.

    Parameters
    ----------
    **kwargs
        Parameters for `dataclasses.field` besides `repr` (with the other ones before
        them for `DataDescriptor`).

    """
    validate = _canonicalize_callables(validate)
    transform = _canonicalize_callables(transform)

    validate = (break_on_jax_placeholder,) + validate
    transform = (break_on_jax_placeholder,) + transform
    if lax.stop_gradient not in transform:
        transform = transform + (lax.stop_gradient,)

    kwargs = _update_metadata(kwargs, FType.FIXED)

    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        repr=repr, **kwargs,
    )


def aux_field(*, mandatory=True, default=None, default_function=None, cache=None,
              validate=None, transform=None, repr=False, **kwargs):
    """Descriptor dataclass field for pytree auxiliary data, which must be hashable.

    `dataclasses.Field.metadata` is updated with ``'ftype'``. `repr` is supppressed by
    default. See `DataDescriptor`, `dataclasses.field`, and JAX pytree documentations.

    Parameters
    ----------
    **kwargs
        Parameters for `dataclasses.field` besides `repr` (with the other ones before
        them for `DataDescriptor`).

    """
    kwargs = _update_metadata(kwargs, FType.AUXILIARY)

    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        repr=repr, **kwargs,
    )


class DataTree(ABC):
    """Base class for combining `DataDescriptor`, `dataclasses.dataclass`, and
    optionally JAX pytree.

    Use it together with either `dataclasses.dataclass` or `pytree_dataclass`.
    `dataclasses.__post_init__` is implemented to check missing mandatory arguments, and
    to compute and fill values using `DataDescriptor.default_function`. Also added are
    pretty string by `pprint.pformat`, a method that `replace` fields with changes, and
    methods to `cache` and `purge` fields.

    Raises
    ------
    TypeError
        If not a `dataclasses.dataclass` at instance creation.

    Examples
    --------
    >>> @dataclasses.dataclass(frozen=True)
    ... class Euler(DataTree):
    ...     e: float = field(default=2.7182818, validate=float, init=False)
    ...     pi: float = field(default=3.1415926, validate=float, init=False)
    ...     i: complex = DataDescriptor(default=1j, validate=complex)
    ...     one: complex = DataDescriptor(default=1, validate=complex)
    ...     zero: complex = DataDescriptor(
    ...         default_function=lambda self: self.e ** (self.i * self.pi) + self.one,
    ...         validate=(abs, float, lambda value: round(value, ndigits=5), complex))
    >>> print(Euler())
    Euler(e=2.7182818, pi=3.1415926, i=1j, one==(1+0j), zero=0j)

    Also see the examples in `pytree_dataclass`.

    """

    def __new__(cls, *args, **kwargs):
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f'{cls.__qualname__} must be a dataclasses.dataclass')
        return super().__new__(cls)

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self):
        return pformat(self)  # python >= 3.10

    def __post_init__(self):
        for field in dataclasses.fields(self):
            descr = vars(type(self)).get(field.name, None)
            if isinstance(descr, DataDescriptor):
                # field(default=DataDescriptor(...), init=False, ...)
                # "broken behavior" in cpython issue #102646
                # TODO add a test for init=False, and then a pointer to that here
                if not field.init:
                    descr.__set__(self, field.default)

                descr.raise_missing(self)

                value = descr.run_default_function(self)
                if value is not self:
                    descr.__set__(self, value)

    def replace(self, **changes):
        """Create a new object of the same type, replacing fields with changes.

        See `dataclasses.replace`.

        """
        return dataclasses.replace(self, **changes)

    def cache(self, *args):
        """Cache specified fields in the order of `args`, ignoring absent or non-caching
        ones.

        Parameters
        ----------
        args
            Names of the fields to cache. Pass a single `...` to cache all fields in the
            order of `dataclasses.fields`.

        Returns
        -------
        obj : DataTree
            A new object with specified fields cached.

        """
        if len(args) == 1 and args[0] is Ellipsis:
            args = (field.name for field in dataclasses.fields(self))
        obj = self
        for name in args:
            descr = vars(type(self)).get(name, None)
            if isinstance(descr, DataDescriptor):
                value = descr.run_cache(obj)
                if value is not obj:
                    obj = obj.replace(**{name: value})
        return obj

    def purge(self, *args):
        """Purge specified fields, ignoring absent or non-caching ones.

        Parameters
        ----------
        args
            Names of the fields to purge. Pass a single `...` to purge all fields.

        Returns
        -------
        obj : DataTree
            A new object with specified fields set to `None`.

        """
        if len(args) == 1 and args[0] is Ellipsis:
            args = (field.name for field in dataclasses.fields(self))
        obj = self
        for name in args:
            descr = vars(type(self)).get(name, None)
            if isinstance(descr, DataDescriptor) and descr.cache is not None:
                obj = obj.replace(**{name: None})
        return obj


# TODO maybe move this and add, sub, etc to something like tan.py
class TanMixin:
    """Addition and scalar multiplication operations for tangent and cotangent vector
    spaces.

    """

    def __add__(self, other):
        return tree_map(add, self, other)

    def __sub__(self, other):
        return tree_map(sub, self, other)

    def __neg__(self):
        return tree_map(neg, self)

    def __mul__(self, scalar):
        return tree_map(partial(scalar_mul, scalar), self)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return tree_map(partial(scalar_div, scalar), self)


def pytree_dataclass(cls, *, frozen=True, **kwargs):
    """Register classes as dataclasses and pytree nodes.

    `iter_fields` is implemented to iterate over fields of selected pytree dataclass
    field type. See the example below for how to combine this with `DataTree`,
    `dyn_field`, `fxd_field`, and `aux_field` for full power.

    Parameters
    ----------
    cls : type
        Class to be registered.
    frozen : bool, optional
        Whether to return a frozen dataclass that emulates read-only behavior, frozen by
        default which one shouldn't need to change.
    **kwargs
        Other parameters besides `frozen` for the `dataclasses.dataclass`.

    Returns
    -------
    cls : type
        Registered pytree dataclass.

    Raises
    ------
    ValueError
        If ``'ftype'`` in `dataclasses.Field.metadata` is not recognized.

    Notes
    -----
    The pytree nomenclature differs from that of the ordinary tree in its definition of
    "node": pytree leaves are not pytree nodes in the JAX documentation. The leaves
    contain data to be traced by JAX transformations, while the nodes are Python
    (including None) and extended containers to be mapped over.

    References
    ----------
    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d

    .. _flax.struct package — Flax documentation:
        https://flax.readthedocs.io/en/latest/flax.struct.html

    .. _Extra features - Equinox:
        https://docs.kidger.site/equinox/api/module/advanced_fields/

    .. _cgarciae/simple-pytree:
        https://github.com/cgarciae/simple-pytree

    .. _JAX Issue #2371:
        https://github.com/google/jax/issues/2371

    Examples
    --------
    >>> @pytree_dataclass
    ... class Parameters(TanMixin, DataTree):
    ...     dtype: DTypeLike = aux_field(default=jnp.complex64,
    ...                                  validate=issubdtype_of(jnp.complexfloating)
    ...                                  repr=True)
    ...     theta: ArrayLike = dyn_field(default=jnp.array([0, 1, 1j]),
    ...                                  validate=asarray_of(field='dtype'))
    ...     const: ArrayLike = fxd_field(default=jnp.array([2.7182818, 3.1415926]),
    ...                                  validate=jnp.float32
    ...                                  repr=True)
    >>> print(Parameters())
    Parameters(dtype=<class 'jax.numpy.complex64'>,
               theta=Array([0.+0.j, 1.+0.j, 0.+1.j], dtype=complex64),
               const=Array([2.7182817, 3.1415925], dtype=float32))

    """
    cls = dataclasses.dataclass(cls, frozen=frozen, **kwargs)

    for field in dataclasses.fields(cls):
        if field.metadata.get('ftype', FType.DYNAMIC) not in FType:
            raise ValueError(f"metadata['ftype']={field.metadata['ftype']} "
                             f'of field {field.name} not recognized')

    def iter_fields(self=None, ftype=FType, name=False, key=False, value=False):
        """Iterate over field names, keys, and/or values, of given `ftype`.

        Parameters
        ----------
        self : DataTree, optional
            Pytree dataclass object, `None` by default, so that we can get names and/or
            keys (but not values) without an object.
        ftype : FType or its members, optional
            Pytree dataclass field type to select.
        name : bool, optional
            Whether to select field name. It's also possible to get names from keys by
            ``key.name``.
        key : bool, optional
            Whether to select pytree key. It's also possible to get names from keys by
            ``key.name``.
        value : bool, optional
            Whether to select field value.

        Raises
        ------
        ValueError
            If none of `name`, `key`, or `value` is selected, or if selecting `value`
            when `self` is `None`.

        """
        if not name and not key and not value:
            raise ValueError('must select at least one among name, key, and value')
        if value and self is None:
            raise ValueError('values unavailable without a pytree dataclass object')

        for field in dataclasses.fields(cls):
            if field.metadata.get('ftype', FType.DYNAMIC) in ftype:
                item = []
                if name:
                    item.append(field.name)
                if key:
                    item.append(GetAttrKey(field.name))
                if value:
                    item.append(getattr(self, field.name))
                yield tuple(item) if len(item) > 1 else item[0]
    cls.iter_fields = iter_fields

    def flatten_with_keys(obj):
        children_with_keys = obj.iter_fields(ftype=FType.CHILD, key=True, value=True)
        aux_data = obj.iter_fields(ftype=FType.AUXILIARY, value=True)
        return tuple(children_with_keys), tuple(aux_data)

    def flatten_func(obj):
        children = obj.iter_fields(ftype=FType.CHILD, value=True)
        aux_data = obj.iter_fields(ftype=FType.AUXILIARY, value=True)
        return tuple(children), tuple(aux_data)

    def unflatten_func(aux_data, children):
        children_names = iter_fields(ftype=FType.CHILD, name=True)
        aux_data_names = iter_fields(ftype=FType.AUXILIARY, name=True)
        return cls(**dict(zip(children_names, children)),
                   **dict(zip(aux_data_names, aux_data)))

    register_pytree_with_keys(cls, flatten_with_keys, unflatten_func, flatten_func)

    return cls
