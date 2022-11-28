import dataclasses
from pprint import pformat

from jax.tree_util import register_pytree_node, tree_leaves


def pytree_dataclass(cls, aux_fields=None, aux_invert=False, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    Also added are methods that return children and aux_data iterators, and pretty
    string representation, and a method that replace fields with changes.

    Parameters
    ----------
    cls : type
        Class to be registered, not a python dataclass yet.
    aux_fields : str, sequence of str, or Ellipsis, optional
        Pytree aux_data fields. Default is none; unrecognized ones are ignored;
        ``Ellipsis`` uses all.
    aux_invert : bool, optional
        Whether to invert ``aux_fields`` selections, convenient when most but not all
        fields are aux_data.
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

    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d

    .. _flax.struct package — Flax documentation:
        https://flax.readthedocs.io/en/latest/flax.struct.html

    .. _JAX Issue #2371:
        https://github.com/google/jax/issues/2371

    """
    if dataclasses.is_dataclass(cls):
        raise TypeError('cls cannot already be a dataclass')
    cls = dataclasses.dataclass(cls, **kwargs)

    if aux_fields is None:
        aux_fields = ()
    elif isinstance(aux_fields, str):
        aux_fields = (aux_fields,)
    elif aux_fields is Ellipsis:
        aux_fields = [field.name for field in dataclasses.fields(cls)]
    aux_data_names = [field.name for field in dataclasses.fields(cls)
                      if field.name in aux_fields]
    children_names = [field.name for field in dataclasses.fields(cls)
                      if field.name not in aux_fields]

    if aux_invert:
        aux_data_names, children_names = children_names, aux_data_names

    def children(self):
        """Return an iterator over pytree children values."""
        for name, value in self.named_children():
            yield value

    def named_children(self):
        """Return an iterator over pytree children names and values."""
        for name in children_names:
            value = getattr(self, name)
            yield name, value

    def aux_data(self):
        """Return an iterator over pytree aux_data values."""
        for name, value in self.named_aux_data():
            yield value

    def named_aux_data(self):
        """Return an iterator over pytree aux_data names and values."""
        for name in aux_data_names:
            value = getattr(self, name)
            yield name, value

    cls.children = children
    cls.named_children = named_children
    cls.aux_data = aux_data
    cls.named_aux_data = named_aux_data

    def tree_flatten(obj):
        #FIXME JAX doesn't like the flatten function to return iterators, and somehow
        #triggered AssertionError by _closure_convert_for_avals in custom_derivatives.py
        return tuple(obj.children()), tuple(obj.aux_data())

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(children_names, children)),
                   **dict(zip(aux_data_names, aux_data)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

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
