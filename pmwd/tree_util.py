import dataclasses
from pprint import pformat

from jax import float0
from jax.numpy import asarray
from jax.tree_util import register_pytree_node


def pytree_dataclass(cls, aux_fields=None, aux_invert=False, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    Also added are pretty string representation and replace method.

    Parameters
    ----------
    cls : type
        Class to be registered, not a python dataclass yet.
    aux_fields : str, sequence of str, or Ellipsis, optional
        Pytree aux_data fields. Default is none; unrecognized ones are ignored;
        ``Ellipsis`` uses all.
    aux_invert : bool, optional
        Whether to invert ``aux_fields`` selections, convenient when most fields are not
        aux_data.
    kwargs
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

    .. _Automatically treat dataclasses as pytrees · Issue #2371 · google/jax:
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
        aux_fields = tuple(field.name for field in dataclasses.fields(cls))
    akeys = tuple(field.name for field in dataclasses.fields(cls)
                  if field.name in aux_fields)
    ckeys = tuple(field.name for field in dataclasses.fields(cls)
                  if field.name not in aux_fields)

    if aux_invert:
        akeys, ckeys = ckeys, akeys

    def tree_flatten(obj):
        children = tuple(getattr(obj, key) for key in ckeys)
        aux_data = tuple(getattr(obj, key) for key in akeys)
        return children, aux_data

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(ckeys, children)), **dict(zip(akeys, aux_data)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    @staticmethod
    def _safe_asarray_factory(dtype=None):
        """Return an array converter that's safe for float0 and JAX transformations.

        .. _Pytrees — JAX documentation:
            https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

        .. _JAX Issues #10238:
            https://github.com/google/jax/issues/10238

        """
        def _safe_asarray(x):
            if not (hasattr(x, 'dtype') and x.dtype == float0
                    or type(x) is object or x is None or isinstance(x, cls)):
                x = asarray(x, dtype=dtype)
            return x

        return _safe_asarray

    setattr(cls, '_safe_asarray_factory', _safe_asarray_factory)

    def __str__(self):
        """Pretty string representation for python >= 3.10."""
        return pformat(self)

    setattr(cls, '__str__', __str__)

    def replace(self, **changes):
        """Create a new object of the same type, replacing fields with changes."""
        return dataclasses.replace(self, **changes)

    setattr(cls, 'replace', replace)

    return cls
