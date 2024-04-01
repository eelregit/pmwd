"""SO input features consists of simple Sobol parameters, a and k etc."""
import jax.numpy as jnp


def sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    theta = jnp.asarray([
        conf.cell_size,
        a,
        cosmo.A_s_1e9,
        cosmo.n_s,
        cosmo.Omega_m,
        cosmo.Omega_b,
        cosmo.Omega_k,
        cosmo.h,
        conf.softening_length,
    ])
    return theta


def soft_names(net):
    # str names of input features of the SO neural nets
    # currently hardcoded, should be updated along with functions above
    theta = ['l_c', 'a', 'A_s_1e9', 'n_s', 'Omega_m', 'Omega_b',
             'Omega_k', 'h', 'l_s']
    if net == 'f':
        theta = ['k'] + theta
    if net == 'g':
        theta = ['k_1', 'k_2', 'k_3'] + theta

    return theta


def soft_names_tex(net):
    # soft_names in latex math expressions
    theta = ['l_c', 'a', 'A_s', 'n_s', '\\Omega_m', '\\Omega_b',
             '\\Omega_k', 'h', 'l_s']
    if net == 'f':
        theta = ['k'] + theta
    if net == 'g':
        theta = ['k_1', 'k_2', 'k_3'] + theta

    return theta


def soft_len(net):
    # get the length of SO input features
    return len(soft_names(net))


def soft(k, theta):
    """SO features for neural nets input, with k being a scalar."""
    return jnp.concatenate((jnp.atleast_1d(k), theta))


def soft_k(k, theta):
    """Get SO input features (k, theta)."""
    theta = jnp.broadcast_to(theta, k.shape+theta.shape)
    ft = jnp.concatenate((k.reshape(k.shape + (1,)), theta), axis=-1)
    return ft


def soft_kvec(kv, theta):
    """Get SO input features (k1, k2, k3, theta)."""
    theta = jnp.broadcast_to(theta, kv.shape[:-1]+theta.shape)
    ft = jnp.concatenate((kv, theta), axis=-1)
    return ft
