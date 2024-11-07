from abc import ABC, abstractmethod
from collections.abc import Callable
import dataclasses
from enum import Flag, auto
from functools import partial
from pprint import pformat

import jax.numpy as jnp
from jax.tree_util import GetAttrKey, register_pytree_with_keys, tree_leaves, tree_map
from jax.lax import stop_gradient

from pmwd.util import add, sub, neg, scalar_mul, scalar_div


class FType(Flag):
    """Pytree dataclass field types."""
    DYNAMIC = auto()
    FIXED = auto()
    AUXILIARY = auto()
    CHILD = DYNAMIC | FIXED
FType.DYNAMIC.__doc__ = 'Dynamic pytree children with gradients.'
FType.FIXED.__doc__ = 'Fixed pytree children without gradients.'
FType.CHILD.__doc__ = 'Pytree children, dynamic or fixed.'
FType.AUXILIARY.__doc__ = 'Auxiliary data stored in pytree treedef.'


# FIXME passing tuple (jnp.dtype, issubdtype_of(jnp.floating)) for dtype validation
def issubdtype_of(dtype):
    def fun(test_dtype):
        if not jnp.issubdtype(test_dtype, dtype):
            raise ValueError(f'{test_dtype!r} must be sub-dtype of {dtype!r}')
        return test_dtype
    return fun


def asarray_of(dtype=None):
    def fun(value):
        return tree_map(partial(jnp.asarray, dtype=dtype), value)
    return fun


def _astuple_of_callables(fun):
    if fun is None:
        fun = ()
    elif isinstance(fun, Callable):
        fun = (fun,)
    else:
        fun = tuple(fun)
    return fun


def _is_jax_placeholder(obj):
    """Whether an object is a placeholder traced by JAX transformations.

    .. _Pytrees — JAX documentation:
        https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

    .. _JAX Issue #10238:
        https://github.com/google/jax/issues/10238

    """
    return type(obj) is object


# FIXME maybe not useful anymore
#def _is_transforming(self):
#    """Whether fields are placeholders traced by JAX transformations.
#
#    .. _Pytrees — JAX documentation:
#        https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
#
#    .. _JAX Issue #10238:
#        https://github.com/google/jax/issues/10238
#
#    """
#    def leaves_all(is_placeholder, tree):
#        # similar to tree_all(tree_map(is_placeholder, tree))
#        return all(is_placeholder(x) for x in tree_leaves(tree))
#
#    # unnecessary to test for None's since they are empty pytree nodes
#    return tree_leaves(self) and leaves_all(_is_jax_placeholder, self)


class DataDescriptor:
    """Descriptor to specify computation rules of described data.

    Parameters
    ----------
    mandatory : bool, optional
        Whether the value must be initialized to something other than ``None``, if none
        of ``default``, ``default_function``, or ``cache`` is specified.
    default : pytree, optional
        Default value.
    default_function : callable or None, optional
        Default function to compute the value, if not already initialized to anything
        else but ``None``, from the instance of the owner class, ``value =
        default_function(obj)``, runned by e.g. ``DataTree.__post_init__``.
    cache : callable or None, optional
        Caching function to compute the value from the instance of the owner class,
        ``value = cache(obj)``, runned by e.g. ``DataTree.cache``.
    validate : callable, sequence of callable, or None, optional
        Validator functions before setting the value, ``value = validate(value)``. If a
        sequence, apply each in turn. Skipped if ``value is None``, or if ``type(value)
        is object`` to avoid validating JAX placeholders.
    transform : callable, sequence of callable, or None, optional
        Transformer functions after getting the value, ``value = transform(value)``. If
        a sequence, apply each in turn. Skipped if ``value is None``, or if
        ``type(value) is object`` to avoid transforming JAX placeholders. Useful with
        `lax.stop_gradient`.

    Methods?
    -------
    FIXME

    Raises
    ------
    ValueError
        If more than one is specified among ``default``, ``default_function``, and
        ``cache``, or if mandatory data is missing when none of the three is specified.
    TypeError
        If trying to set or delete descriptor attributes.

    .. _Python Descriptor HowTo Guide:
        https://docs.python.org/3/howto/descriptor.html

    .. _Python Data Model - Implementing Descriptors:
        https://docs.python.org/3/reference/datamodel.html#implementing-descriptors

    .. _Python Data Model - Invoking Descriptors:
        https://docs.python.org/3/reference/datamodel.html#invoking-descriptors

    .. _Python Data Classes:
        https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields

    .. _Combining a descriptor class with dataclass and field:
        https://stackoverflow.com/questions/67612451/combining-a-descriptor-class-with-dataclass-and-field

    .. _Google etils dataclass field descriptor:
        https://github.com/google/etils/blob/main/etils/edc/field_utils.py

    .. _FIXME Dataclass descriptor behavior inconsistent:
        https://github.com/python/cpython/issues/102646

    """

    __slots__ = (
        'mandatory',
        'default',
        'default_function',
        'cache',
        'validate',
        'transform',
        '_name',
    )

    def __init__(self, mandatory=True, default=None, default_function=None, cache=None,
                 validate=None, transform=None):
        if sum(x is not None for x in (default, default_function, cache)) > 1:
            raise ValueError(f'{default=}, {default_function=}, and {cache=} are '
                             'mutually exclusive')

        validate = _astuple_of_callables(validate)
        transform = _astuple_of_callables(transform)

        object.__setattr__(self, 'mandatory', mandatory)
        object.__setattr__(self, 'default', default)
        object.__setattr__(self, 'default_function', default_function)
        object.__setattr__(self, 'cache', cache)
        object.__setattr__(self, 'validate', validate)
        object.__setattr__(self, 'transform', transform)

    def __repr__(self):
        return (
            f'{__class__.__name__}('
            f'mandatory={self.mandatory}, '
            f'default={self.default!r}, '
            f'default_function={self.default_function!r}, '
            f'cache={self.cache!r}, '
            f'validate={self.validate!r}, '
            f'transform={self.transform!r})'
        )

    def __setattr__(self, name, value):
        raise TypeError(f'{self.__class__} is read-only')

    def __delattr__(self, name):
        raise TypeError(f'{self.__class__} is read-only')

    def __set_name__(self, objtype, name):
        object.__setattr__(self, '_name', '_' + name)

    def __get__(self, obj, objtype=None):
        #print(f'__get__: {type(obj)=!r}, {obj=!r}', flush=True)
        #print(f'__get__: {type(objtype)=!r}, {objtype=!r}', flush=True)
        if obj is None:  # name: type = descr
            #print(f'__get__: obj is None, {objtype=!r}', flush=True)
            return self.default
        value = getattr(obj, self._name)
        return self.run_transform(value)

    def __set__(self, obj, value):
        #try:
        #    print(f'__set__: {type(obj)=!r}, {obj=!r}', flush=True)
        #except AttributeError:
        #    print(f'__set__: {type(obj)=!r}, CANNOT print(obj)', flush=True)
        #print(f'__set__: {type(value)=!r}, {value=!r}', flush=True)
        if value is self:  # name: type = field(default=descr, ...)
            #print(f'__set__: {value=!r} is self', flush=True)
            value = self.default
        value = self.run_validate(value)
        object.__setattr__(obj, self._name, value)

    def raise_missing(self, obj):
        if self.mandatory and all(x is None for x in (
                self.default, self.default_function, self.cache, self.__get__(obj))):
            raise ValueError('mandatory data missing')

    def run_default_function(self, obj):
        if self.default_function is None or self.__get__(obj) is not None:
            return obj
        value = self.default_function(obj)
        return value

    def run_cache(self, obj):
        if self.cache is None:
            return obj
        value = self.cache(obj)
        return value

    def run_validate(self, value):
        if value is None or _is_jax_placeholder(value):
            return value
        for validate in self.validate:
            value = validate(value)
        return value

    def run_transform(self, value):
        if value is None or _is_jax_placeholder(value):
            return value
        for transform in self.transform:
            value = transform(value)
        return value


def _forbid_default_factory(kwargs):
    if 'default_factory' in kwargs:
        raise ValueError('default_factory not supported')


def _update_metadata(kwargs, ftype):
    metadata = kwargs.pop('metadata', {})
    metadata = {} if metadata is None else dict(metadata)
    metadata['ftype'] = ftype
    kwargs['metadata'] = metadata
    return kwargs


def dyn_field(*, mandatory=True, default=None, default_function=None, cache=None,
              validate=None, transform=None, **kwargs):
    """Descriptor dataclass field for dynamic pytree children.

    ``metadata`` is updated with ``ftype``. ``default_factory`` is not supported. See
    ``DataDescriptor`` and ``dataclasses.field``.

    Raises
    ------
    ValueError
        If ``default_factory`` is specified.

    """
    _forbid_default_factory(kwargs)
    kwargs = _update_metadata(kwargs, FType.DYNAMIC)
    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        **kwargs,
    )


def fxd_field(*, mandatory=True, default=None, default_function=None, cache=None,
              validate=None, transform=None, repr=False, **kwargs):
    """Descriptor dataclass field for fixed pytree children.

    ``lax.stop_gradient`` is appended to ``transform`` if not already in it.
    ``metadata`` is updated with ``ftype``. ``default_factory`` is not supported.
    ``repr`` is supppressed by default. See ``DataDescriptor`` and
    ``dataclasses.field``.

    Raises
    ------
    ValueError
        If ``default_factory`` is specified.

    """
    _forbid_default_factory(kwargs)
    kwargs = _update_metadata(kwargs, FType.FIXED)
    transform = _astuple_of_callables(transform)
    if stop_gradient not in transform:
        transform += (stop_gradient,)
    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        repr=repr, **kwargs,
    )


def aux_field(*, mandatory=True, default=None, default_function=None, cache=None,
              validate=None, transform=None, repr=False, **kwargs):
    """Descriptor dataclass field for auxiliary data.

    ``metadata`` is updated with ``ftype``. ``default_factory`` is not supported.
    ``repr`` is supppressed by default. See ``DataDescriptor`` and
    ``dataclasses.field``.

    Raises
    ------
    ValueError
        If ``default_factory`` is specified.

    """
    _forbid_default_factory(kwargs)
    kwargs = _update_metadata(kwargs, FType.AUXILIARY)
    return dataclasses.field(
        default=DataDescriptor(mandatory, default, default_function, cache,
                               validate, transform),
        repr=repr, **kwargs,
    )


class DataTree(ABC):
    """Base class for combining pytree and dataclass.

    Linear operators (addition, subtraction, and scalar multiplication) are defined for
    tangent and cotangent vector spaces.

    FIXME add idiom to pytree_dataclass below

    """

    @abstractmethod
    def __init__(self):
        pass

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

    def __str__(self):
        """Pretty string representation for python >= 3.10."""
        return pformat(self)

    def __post_init__(self):
        for name in self.iter_fields(name=True):
            descr = vars(self.__class__).get(name, None)
            if isinstance(descr, DataDescriptor):
                descr.raise_missing(self)
                value = descr.run_default_function(self)
                if value is not self:
                    descr.__set__(self, value)

    def replace(self, **changes):
        """Create a new object of the same type, replacing fields with changes.

        See ``dataclasses.replace``.

        """
        return dataclasses.replace(self, **changes)

    def cache(self):
        """Cache all fields in the order of ``dataclasses.fields``.

        Returns
        -------
        instance : DataTree
            A new instance with all fields cached.

        """
        obj = self
        for name in self.iter_fields(name=True):
            descr = vars(self.__class__).get(name, None)
            if isinstance(descr, DataDescriptor):
                value = descr.run_cache(obj)
                if value is not obj:
                    obj = obj.replace(**{name: value})
        return obj

    def purge(self):
        """Purge all caches.

        Returns
        -------
        instance : DataTree
            A new instance with all cached fields set to ``None``.

        """
        obj = self
        for name in self.iter_fields(name=True):
            descr = vars(self.__class__).get(name, None)
            if isinstance(descr, DataDescriptor) and descr.cache is not None:
                obj = obj.replace(**{name: None})
        return obj

    # FIXME no need to leave things as is now, so this API may not be the best
    def cache_purge(self, **kwargs):
        """Cache and/or purge specified fields in the order of ``kwargs``.

        Parameters
        ----------
        kwargs
            Whether to cache a field, leave it as is (default), or set it to ``None``,
            by passing ``name=True``, ``False``, or ``None``, respectively.

        Returns
        -------
        instance : DataTree
            A new instance containing changed field values.

        Raises
        ------
        ValueError
            If ``kwargs`` has any unrecognized field name or flag value.

        """
        field_names = tuple(self.iter_fields(name=True))
        obj = self

        for name, flag in kwargs.items():
            if name not in field_names:
                raise ValueError('{name} not in fields: {field_names}')
            if not isinstance(flag, bool) and flag is not None:
                raise ValueError('{name}={flag}, must be bool or None')

            if not flag:
                if flag is None:
                    obj = obj.replace(**{name: None})
                continue

            descr = vars(self.__class__).get(name, None)
            if isinstance(descr, DataDescriptor):
                value = descr.run_cache(obj)
                if value is not obj:
                    obj = obj.replace(**{name: value})

        return obj


def pytree_dataclass(cls, *, frozen=True, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    FIXME here and in DataTree
    Also added are methods that generate pytree data (dynamic and fixed) and auxiliary
    data iterators, pretty string representation, and a method that replace fields with
    changes.

    * field or dyn_field marks dynamic pytree children
    * fxd_field marks fixed pytree children
    * aux_field marks auxiliary data
    Data in the dynamic fields (``dyn_field``) are pytree children (``children``), and
    so are data in the fixed fields but with stopped gradients (``fxd_field``). The
    auxiliary fields are not for pytree children but some hashable auxiliary data to be
    stored in the treedef (``aux_data``).

    Auxiliary data are static configurations that are not traced by JAX transformations,
    and their hash values are used to cache the transformed functions. Those not marked
    as auxiliary data fields are pytree children, of which there are three types. The
    fixed pytree children ending with two trailing underscores, e.g., ``bar__``, do not
    receive gradients. These are constants that we don't want to treat as static
    configurations. Some of them set the fixed values of the deactivated optional pytree
    children. The optional pytree children ending with one trailing underscore, e.g.,
    ``foo_``, are extensions when activated. They are None and inactive by default,
    taking the fixed values set by the fixed pytree children ``foo__``. Both the fixed
    and optional pytree children should be accessed through their corresponding
    properties named without the trailing underscores, i.e., ``foo`` and ``bar``. The
    remaining and most common type is the dynamic pytree children, data that are traced
    by JAX and receive gradients.

    Parameters
    ----------
    cls : type
        Class to be registered, not a python dataclass yet.
    frozen : bool, optional
        Whether to return a frozen dataclass that emulates read-only behavior. The
        default has been flipped and you shouldn't need to change this.
    **kwargs
        Other parameters (except ``frozen``) for the Python dataclass decorator.

    Returns
    -------
    cls : type
        Registered dataclass.

    Raises
    ------
    ValueError
        If any field has unrecognized ``ftype`` in metadata.

    Notes
    -----
    The pytree nomenclature differs from that of the ordinary tree in its definition of
    "node": pytree leaves are not pytree nodes in the JAX documentation. The leaves
    contain data to be traced by JAX transformations, while the nodes are Python
    (including None) and extended containers to be mapped over.

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

    """
    cls = dataclasses.dataclass(cls, frozen=frozen, **kwargs)

    for field in dataclasses.fields(cls):
        if field.metadata.get('ftype', FType.DYNAMIC) not in FType:
            raise ValueError("unrecognized metadata['ftype']"
                             f" = {field.metadata['ftype']!r} in field {field.name}")

    def iter_fields(self=None, ftype=FType, name=False, key=False, value=False):
        """Iterate over field names, keys, and/or values, of given ``ftype``.

        Parameters
        ----------
        self : DataTree, optional
            Pytree dataclass instance, ``None`` by default, so that we can get names
            and/or keys (but not values) without an instance.
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
            If none of ``name``, ``key``, or ``value`` is selected, or if selecting
            ``value`` when ``self is None``.

        """
        if not name and not key and not value:
            raise ValueError('must select at least one among name, key, and value')
        if value and self is None:
            raise ValueError('values unavailable without an instance')

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
