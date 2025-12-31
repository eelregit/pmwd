import jax
from jax import pmap
from jax.lax import pmean
from jax.tree_util import tree_map
import jax.numpy as jnp

from datetime import datetime


@pmap(axis_name='device')
def _global_mean(x):
    return pmean(x, axis_name='device')


def arr_global_mean(x):
    """Global average of array across multi processes, using pmap and pmean."""
    # add leading in_axes for pmap over local device within current process
    x = jnp.expand_dims(x, axis=0)
    x = _global_mean(x)
    return x[0]  # rm leading axis, i.e. pmap out_axes


def tree_global_mean(tree):
    """Global average of a pytree, i.e. for all leaves."""
    return tree_map(arr_global_mean, tree)


def procinfo(s, procid, flush=False):
    print(f"[{datetime.now().strftime('%H:%M:%S  %m-%d')}] Proc {procid:>2d}: {s}",
          flush=flush)


def device_sync(procid, n_procs, verbose=False):
    """Nothing but to sync all devices, with dummy global mean."""
    x = tree_global_mean(jnp.array(procid))
    assert round(2 * x + 1) == n_procs, 'something wrong with global mean'
    if verbose:
        procinfo(f'# global devices: {len(jax.devices())}, sync successful', procid,
                 flush=True)
