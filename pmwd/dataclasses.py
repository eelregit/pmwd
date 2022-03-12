from dataclasses import is_dataclass, dataclass, fields

from jax.tree_util import register_pytree_node


def pytree_dataclass(cls, aux_fields=None, aux_invert=False, **kwargs):
    """Register python dataclasses as custom pytree nodes.

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
    if is_dataclass(cls):
        raise TypeError('cls cannot already be a dataclass')
    cls = dataclass(cls, **kwargs)

    if aux_fields is None:
        aux_fields = ()
    elif isinstance(aux_fields, str):
        aux_fields = (aux_fields,)
    elif aux_fields is Ellipsis:
        aux_fields = tuple(field.name for field in fields(cls))
    akeys = tuple(field.name for field in fields(cls) if field.name in aux_fields)
    ckeys = tuple(field.name for field in fields(cls) if field.name not in aux_fields)

    if aux_invert:
        akeys, ckeys = ckeys, akeys

    def tree_flatten(obj):
        children = tuple(getattr(obj, key) for key in ckeys)
        aux_data = tuple(getattr(obj, key) for key in akeys)
        return children, aux_data

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(ckeys, children)), **dict(zip(akeys, aux_data)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    return cls
