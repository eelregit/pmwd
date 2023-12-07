import dataclasses
from pprint import pformat

from jax.tree_util import GetAttrKey, register_pytree_with_keys, tree_leaves
from jax.lax import stop_gradient


def pytree_dataclass(cls, aux_fields=None, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    Also added are methods that return pytree data (dynamic, optional, and fixed) and
    auxiliary data iterators, pretty string representation, and a method that replace
    fields with changes.

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
    aux_fields : str, iterable of str, or Ellipsis, optional
        Auxiliary data fields. Default is none; unrecognized ones are ignored;
        recognized ones must not end with trailing underscores; ``Ellipsis`` uses all.
    **kwargs
        Keyword arguments to be passed to python dataclass decorator.

    Returns
    -------
    cls : type
        Registered dataclass.

    Raises
    ------
    TypeError
        If cls is already a python dataclass.
    ValueError
        If a field name has unexpected trailing underscores.

    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d

    .. _flax.struct package — Flax documentation:
        https://flax.readthedocs.io/en/latest/flax.struct.html

    .. _JAX Issue #2371:
        https://github.com/google/jax/issues/2371

    Notes
    -----
    The pytree nomenclature differs from that of the ordinary tree in its definition of
    "node": pytree leaves are not pytree nodes in the JAX documentation. The leaves
    contain data to be traced by JAX transformations, while the nodes are Python
    (including None) and extended types to be mapped over.

    """
    if dataclasses.is_dataclass(cls):
        raise TypeError(f'cls={cls} must not already be a dataclass')
    cls = dataclasses.dataclass(cls, **kwargs)

    if aux_fields is None:
        aux_fields = ()
    elif isinstance(aux_fields, str):
        aux_fields = (aux_fields,)
    elif aux_fields is Ellipsis:
        aux_fields = tuple(field.name for field in dataclasses.fields(cls))

    dyn_data_names, opt_data_names, fxd_data_names, aux_data_names = [], [], [], []
    for field in dataclasses.fields(cls):
        name = field.name
        if name in aux_fields:
            if name.endswith('_'):
                raise ValueError(f'{name} must not end with trailing underscore')
            aux_data_names.append(name)
        elif name.endswith('___'):
            raise ValueError(f'{name} with >2 trailing underscores not supported')
        elif name.endswith('__'):
            fxd_data_names.append(name)
        elif name.endswith('_'):
            opt_data_names.append(name)
        else:
            dyn_data_names.append(name)
    children_names = dyn_data_names + opt_data_names + fxd_data_names

    def opt_property(name_):
        @property
        def foo(self):
            foo_ = getattr(self, name_)
            foo__ = getattr(self, name_ + '_')
            return stop_gradient(foo__) if foo_ is None else foo_
        return foo

    for name_ in opt_data_names:
        setattr(cls, name_.rstrip('_'), opt_property(name_))

    def fxd_property(name__):
        @property
        def bar(self):
            bar__ = getattr(self, name__)
            return stop_gradient(bar__)
        return bar

    for name__ in fxd_data_names:
        if name__[:-1] not in opt_data_names:
            setattr(cls, name__.rstrip('_'), fxd_property(name__))

    def dyn_data_with_keys(self):
        """Return an iterator over dynamic pytree data with keys."""
        for name in dyn_data_names:
            value = getattr(self, name)
            yield GetAttrKey(name), value

    def dyn_data(self):
        """Return an iterator over dynamic pytree data."""
        for key, value in self.dyn_data_with_keys():
            yield value

    def opt_data_with_keys(self):
        """Return an iterator over optional pytree data with keys."""
        for name in opt_data_names:
            value = getattr(self, name)
            yield GetAttrKey(name), value

    def opt_data(self):
        """Return an iterator over optional pytree data."""
        for key, value in self.opt_data_with_keys():
            yield value

    def fxd_data_with_keys(self):
        """Return an iterator over fixed pytree data with keys."""
        for name in fxd_data_names:
            value = getattr(self, name)
            yield GetAttrKey(name), value

    def fxd_data(self):
        """Return an iterator over fixed pytree data."""
        for key, value in self.fxd_data_with_keys():
            yield value

    def children_with_keys(self):
        """Return an iterator over all pytree data with keys."""
        for name in children_names:
            value = getattr(self, name)
            yield GetAttrKey(name), value

    def children(self):
        """Return an iterator over all pytree data."""
        for key, value in self.children_with_keys():
            yield value

    def aux_data_with_names(self):
        """Return an iterator over auxiliary data with names."""
        for name in aux_data_names:
            value = getattr(self, name)
            yield name, value

    def aux_data(self):
        """Return an iterator over auxiliary data."""
        for name, value in self.aux_data_with_names():
            yield value

    cls.dyn_data_with_keys = dyn_data_with_keys
    cls.dyn_data = dyn_data
    cls.opt_data_with_keys = opt_data_with_keys
    cls.opt_data = opt_data
    cls.fxd_data_with_keys = fxd_data_with_keys
    cls.fxd_data = fxd_data
    cls.children_with_keys = children_with_keys
    cls.children = children
    cls.aux_data_with_names = aux_data_with_names
    cls.aux_data = aux_data

    def tree_flatten_with_keys(obj):
        return list(obj.children_with_keys()), list(obj.aux_data())

    def tree_flatten(obj):
        return list(obj.children()), list(obj.aux_data())

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(children_names, children)),
                   **dict(zip(aux_data_names, aux_data)))

    register_pytree_with_keys(cls, tree_flatten_with_keys, tree_unflatten,
                              flatten_func=tree_flatten)

    def _is_transforming(self):
        """Whether dataclass fields are pytrees initialized by JAX transformations.

        .. _Pytrees — JAX documentation:
            https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

        .. _JAX Issue #10238:
            https://github.com/google/jax/issues/10238

        """
        def leaves_all(is_placeholder, tree):
            # similar to tree_all(tree_map(is_placeholder, tree))
            return all(is_placeholder(x) for x in tree_leaves(tree))

        # unnecessary to test for None's since they are empty pytree nodes
        return tree_leaves(self) and leaves_all(lambda x: type(x) is object, self)

    cls._is_transforming = _is_transforming

    def __str__(self):
        """Pretty string representation for python >= 3.10."""
        return pformat(self)

    cls.__str__ = __str__

    def replace(self, **changes):
        """Create a new object of the same type, replacing fields with changes."""
        return dataclasses.replace(self, **changes)

    cls.replace = replace

    return cls
