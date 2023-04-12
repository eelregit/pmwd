import os
import errno
import h5py
import numpy as np
import jax.numpy as jnp
from jax import device_get

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology
from pmwd.particles import Particles


def read_gadget_hdf5(base, pmid_dtype=jnp.int16, verbose=False):
    """Read dark matter only Gadget HDF5 snapshot or initial condition files.

    Parameters
    ----------
    base : str
        Base of input path. Single file input is read from base.hdf5, and multiple ones
        from base.0.hdf5, base.1.hdf5, etc.
    pmid_dtype : dtype_like, optional
        Signed integer dtype for particle or mesh grid indices.
    verbose : bool, optional
        True for printing more information.

    Raises
    ------
    FileNotFoundError
        If base.hdf5 and base.0.hdf5 not found.
    ValueError
        If number of particles is not a cubic number.

    """
    files = []
    if os.path.exists(fn := f'{base}.hdf5'):
        files.append(fn)
    else:
        i = 0
        while os.path.exists(fn := f'{base}.{i}.hdf5'):
            files.append(fn)
            i += 1
    if not files:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), f'{base}.hdf5 or {base}.0.hdf5')
    if verbose:
        print(f'{len(files)} files:', *files, sep='\n')

    with h5py.File(files[0], 'r') as f:
        header = f['Header'].attrs
        a = header['Time']
        # box_size = header['BoxSize']
        # ptcl_num = header['NumPart_Total'][1]

        # param = f['Parameters'].attrs
        # a_start = param['TimeBegin']
        # a_stop = param['TimeMax']
        # L_cm = param['UnitLength_in_cm']
        # M_g = param['UnitMass_in_g']
        # V_cm_s = param['UnitVelocity_in_cm_per_s']
        # H_0 = param['Hubble']
        # G = param['GravityConstantInternal']
        # Omega_m = param['Omega0']
        # Omega_de = param['OmegaLambda']
        # Omega_b = param['OmegaBaryon']
        # h = param['HubbleParam']

    # ptcl_grid_shape = round(ptcl_num**(1/3))
    # if ptcl_grid_shape**3 != ptcl_num:
    #     raise ValueError(f'number of particles {ptcl_num} is not cubic')

    # conf = Configuration(
    #     ptcl_spacing=box_size / ptcl_grid_shape,
    #     ptcl_grid_shape=(ptcl_grid_shape,) * 3,
    #     pmid_dtype=pmid_dtype,
    #     a_start=a_start,
    #     a_stop=a_stop,
    #     M=M_g / 1e3,
    #     L=L_cm / 1e2,
    #     T=(L_cm / 1e2) / (V_cm_s / 1e2),
    # )

    # Gadget snapshot has no A_s or n_s information
    # cosmo = Cosmology(
    #     conf,
    #     A_s_1e9 = ,
    #     n_s = ,
    #     Omega_m = Omega_m,
    #     Omgea_b = Omega_b,
    #     h = h,
    #     Omega_k_ = 1 - (Omega_m + Omega_de)
    # )

    pos, vel, ids = [], [], []
    for file in files:
        with h5py.File(file, 'r') as f:
            pos.append(f['PartType1']['Coordinates'][:])
            vel.append(f['PartType1']['Velocities'][:])
            ids.append(f['PartType1']['ParticleIDs'][:])
    pos = np.vstack(pos)
    vel = np.vstack(vel) * a**1.5
    ids = np.hstack(ids)

    # the order of Gadget particle ids could change, so need to sort
    # such that the order of particles match that of the ic, i.e. pmwd
    ids = np.argsort(ids)
    pos = pos[ids]
    vel = vel[ids]
    # to convert the pos to pmwd disp, we need to know the mesh shape
    # and the pmid of Lagrangian particle grid;
    # cannot simply use Particles.from_pos, where disp would be wrong;
    # thus we will do this during training;

    return pos, vel, a


def write_gadget_hdf5(base, a, ptcl, cosmo, conf, num_files=1,
                      ids_dtype=np.uint64, float_dtype=np.float64):
    """Write minimal Gadget HDF5 snapshot or initial condition files for dark matter
    only simulations.

    Parameters
    ----------
    base : str
        Base of output path. Single file output is written to base.hdf5, and multiple
        ones to base.0.hdf5, base.1.hdf5, etc.
    a : float
        Scale factor.
    ptcl : Particles
    cosmo : Cosmology
    conf : Configuration
    num_files : int, optional
        Number of output files.
    ids_dtype : dtype_like, optional
        Particle ID dtype for Gadget.
    float_dtype : dtype_like, optional
        Particle float dtype for Gadget.

    Raises
    ------
    ValueError
        If box is not a 3D cube.

    """
    if conf.dim != 3:
        raise ValueError(f'dim={conf.dim} not supported')
    if len(set(conf.box_size)) != 1:
        raise ValueError('noncubic box not supported')

    ids = 1 + ptcl.raveled_id(dtype=ids_dtype)
    pos = ptcl.pos(dtype=float_dtype)
    vel = ptcl.vel.astype(float_dtype) / a**1.5
    pos, vel = device_get([pos, vel])

    # PartType0 is reserved for gas particles in Gadget
    ptcl_num = np.array([0, conf.ptcl_num], dtype=ids_dtype)
    ptcl_mass = np.array([0, cosmo.ptcl_mass], dtype=float_dtype)

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
