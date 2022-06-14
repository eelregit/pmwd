import numpy as np
import h5py

from pmwd.particles import ptcl_pos


def write_gadget_hdf5(
        filename_base,
        num_files,
        ptcl,
        cosmo,
        conf,
        time,
        dtype='f8',
):
    """
    Writes minimal Gadget4 HDF5 initial condition or snapshot files
    for dark matter only simulations.

    Parameters
    ----------
    filename_base : str
        Output is written to filename_base.hdf5 for single file output or
        filename_base.0.hdf5, filename_base.1.hdf5, etc., for multiple files.
    num_files : int
        The number of output files.
    ptcl : Particles
        Particle state.
    cosmo : Cosmology
        Cosmology parameters.
    conf : Configuration
        Configuration parameters.
    time : float
        The scale factor a.
    """
    ntypes = 2
    num_part_total = np.zeros(ntypes, dtype='u4')
    num_part_total[1] = conf.ptcl_num
    num_part_per_file = conf.ptcl_num // num_files

    masses = np.zeros(ntypes, dtype='f8')
    masses[1] = (conf.rho_crit * cosmo.Omega_m * conf.box_vol
                 / num_part_total[1])

    header = {
        'BoxSize': conf.box_size[0],
        'MassTable': masses,
        'NumPart_Total': num_part_total,
        'NumFilesPerSnapshot': num_files,
        'Redshift': 1./time - 1.,
        'Time': time,
    }

    pos = np.asarray(ptcl_pos(ptcl, conf))
    vel = np.asarray(ptcl.vel) * np.power(time, -1.5)
    # ids = np.asarray(ptcl.pmid)
    ids = np.arange(1, conf.ptcl_num+1)

    for i, (part_pos, part_vel, part_ids) in enumerate(zip(
        np.array_split(pos, num_files),
        np.array_split(vel, num_files),
        np.array_split(ids, num_files))):
        if num_files == 1:
            filename = filename_base + '.hdf5'
        else:
            filename = filename_base + '.' + str(i) + '.hdf5'

        num_part_this_file = np.zeros_like(num_part_total)
        num_part_this_file[1] = num_part_per_file
        if i < conf.ptcl_num % num_files:
            num_part_this_file[1] += 1

        with h5py.File(filename, 'w') as f:
            f.create_group('Header')
            for k, v in header.items():
                f['Header'].attrs[k] = v
            f['Header'].attrs['NumPart_ThisFile'] = num_part_this_file

            f.create_group('PartType1')
            f['PartType1'].create_dataset('Coordinates', dtype=dtype,
                                          data=part_pos)
            f['PartType1'].create_dataset('Velocities', dtype=dtype,
                                          data=part_vel)
            f['PartType1'].create_dataset('ParticleIDs', dtype='u4',
                                          data=part_ids)
