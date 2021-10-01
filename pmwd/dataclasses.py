from dataclasses import is_dataclass, dataclass, fields

from jax.tree_util import register_pytree_node


def pytree_dataclass(cls):
    """Decorator for python dataclasses as custom pytree nodes

    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d
    """
    if not is_dataclass(cls):
        cls = dataclass(cls)

    keys = [field.name for field in fields(cls)]

    def tree_flatten(obj):
        children = [getattr(obj, key) for key in keys]
        aux_data = None
        return children, aux_data

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(keys, children)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    return cls
