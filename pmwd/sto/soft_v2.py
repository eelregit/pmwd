"""SO input features consists of simple Sobol parameters, a and k etc."""
import jax.numpy as jnp


# exp(ln(k*scale)) seems to always appear in SR expressions
# log might be unnecessary for NN training
log_k_theta = False


def sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    # quantities of dim L
    theta_l = jnp.asarray([
        conf.cell_size,
        conf.softening_length,
    ])

    # dimensionless quantities
    theta_o = jnp.asarray([
        a,
        cosmo.A_s_1e9,
        cosmo.n_s,
        cosmo.Omega_m,
        cosmo.Omega_b,
        cosmo.Omega_k,
        cosmo.h,
    ])
    return (theta_l, theta_o)


def soft_names(net):
    # str names of input features of the SO neural nets
    # currently hardcoded, should be updated along with functions above
    theta_l = ['l_c', 'l_s']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k{v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n}{v}')
    if log_k_theta:
        theta_l_k = [f'log({v})' for v in theta_l_k]

    theta_o = ['a', 'A_s_1e9', 'n_s', 'Omega_m', 'Omega_b', 'Omega_k', 'h']

    return theta_l_k + theta_o


def soft_names_tex(net):
    # soft_names in latex math expressions
    theta_l = ['l_c', 'l_s']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k {v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n} {v}')
    if log_k_theta:
        theta_l_k = [f'\\ln {v}' for v in theta_l_k]

    theta_o = ['a', 'A_s', 'n_s', '\\Omega_m', '\\Omega_b', '\\Omega_k', 'h']

    return theta_l_k + theta_o


def soft_len(net):
    # get the length of SO input features
    return len(soft_names(net))


def soft(k, theta):
    """SO features for neural nets input, with k being a scalar."""
    theta_l, theta_o = theta
    k_theta_l = k * theta_l
    if log_k_theta:
        k_theta_l = jnp.log(jnp.where(k_theta_l > 0., k_theta_l, 1.))
    return jnp.concatenate((k_theta_l, theta_o))


def soft_k(k, theta):
    """Get SO input features (k * l, o)."""
    theta_l, theta_o = theta  # e.g. (8,), (6,)
    k_shape = k.shape  # e.g. (128, 1, 1)
    k = k.reshape(k_shape + (1,))  # (128, 1, 1, 1)
    theta_l = theta_l.reshape((1,) * len(k_shape) + theta_l.shape)  # (1, 1, 1, 8)
    ft = k * theta_l  # (128, 1, 1, 8)
    if log_k_theta:
        ft = jnp.log(jnp.where(ft > 0., ft, 1.))
    theta_o = jnp.broadcast_to(theta_o, k_shape+theta_o.shape)  # (128, 1, 1, 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (128, 1, 1, 8+6)
    return ft


def soft_kvec(kv, theta):
    """Get SO input features (k1 * l, k2 * l, k3 * l, o)."""
    kv_shape = kv.shape  # e.g. (128, 128, 65, 3)
    kv = kv.reshape(kv_shape + (1,))  # (128, 128, 65, 3, 1)

    theta_l, theta_o = theta  # e.g. (8,), (6,)
    theta_l = theta_l.reshape((1,) * len(kv_shape) + theta_l.shape)  # (1, 1, 1, 1, 8)
    ft = kv * theta_l  # (128, 128, 65, 3, 8)
    ft = ft.reshape(kv_shape[:-1] + (-1,))  # (128, 128, 65, 3*8)
    if log_k_theta:
        ft = jnp.log(jnp.where(ft > 0., ft, 1.))
    theta_o = jnp.broadcast_to(theta_o, kv_shape[:-1]+theta_o.shape)  # (128, 128, 65, 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (128, 128, 65, 3*8+6)
    return ft
