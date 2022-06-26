import h5py
import numpy as np
import jax.numpy as jnp
from jax import device_get

from pmwd.particles import ptcl_pos


def read_gadget_hdf5(base, pmid_dtype=jnp.int16):
    """Read dark-matter-only Gadget HDF5 snapshot or initial condition files.

    Parameters
    ----------
    base : str
        Base of input path. Single file input is read from base.hdf5, and multiple ones
        from base.0.hdf5, base.1.hdf5, etc.
    pmid_dtype : dtype_like, optional
        Signed integer dtype for particle or mesh grid indices.

    Returns
    -------
    a : float
        Scale factor.
    ptcl : Particles
    cosmo : Cosmology
    conf : Configuration

    """
    raise NotImplementedError


def write_gadget_hdf5(base, num_files, a, ptcl, cosmo, conf, ids_dtype=np.uint32):
    """Write minimal Gadget HDF5 snapshot or initial condition files for dark matter
    only simulations.

    Parameters
    ----------
    base : str
        Base of output path. Single file output is written to base.hdf5, and multiple
        ones to base.0.hdf5, base.1.hdf5, etc.
    num_files : int
        Number of output files.
    a : float
        Scale factor.
    ptcl : Particles
    cosmo : Cosmology
    conf : Configuration
    ids_dtype : dtype_like, optional
        Particle ID precision for Gadget.

    Raises
    ------
    ValueError
        If box is not a 3D cube.

    """
    if conf.dim != 3:
        raise ValueError('dim={conf.dim} not supported')
    if len(set(conf.box_size)) != 1:
        raise ValueError('noncubic box not supported')

    ids = np.arange(1, 1+conf.ptcl_num, dtype=ids_dtype)
    pos = ptcl_pos(ptcl, conf).astype(conf.float_dtype)
    vel = ptcl.vel.astype(conf.float_dtype) / a**1.5
    pos, vel = device_get([pos, vel])

    # PartType0 is reserved for gas particles in Gadget
    ptcl_num = np.array([0, conf.ptcl_num], dtype=ids_dtype)
    ptcl_mass = np.array([0, cosmo.ptcl_mass], dtype=conf.float_dtype)

    header = {
        'BoxSize': conf.box_size[0],
        'MassTable': ptcl_mass,
        'NumFilesPerSnapshot': num_files,
        'NumPart_Total': ptcl_num,
        'Redshift': 1/a - 1,
        'Time': a,
    }

    for i, (ids_chunk, pos_chunk, vel_chunk) in enumerate(zip(
        np.array_split(ids, num_files, axis=0),
        np.array_split(pos, num_files, axis=0),
        np.array_split(vel, num_files, axis=0))):

        suffix = '.hdf5'
        if num_files != 1:
            suffix = f'.{i}' + suffix
        path = base + suffix

        ptcl_num = np.array([0, len(ids_chunk)], dtype=ids_dtype)

        with h5py.File(path, 'w') as f:
            h = f.create_group('Header')
            for k, v in header.items():
                h.attrs[k] = v
            h.attrs['NumPart_ThisFile'] = ptcl_num

            g = f.create_group('PartType1')
            g.create_dataset('ParticleIDs', data=ids_chunk)
            g.create_dataset('Coordinates', data=pos_chunk)
            g.create_dataset('Velocities', data=vel_chunk)
