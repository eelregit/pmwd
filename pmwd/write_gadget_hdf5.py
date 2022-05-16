import numpy as np
from h5py import File


def writeGadgetHDF5(
        filename_base,
        num_files,
        pos, vel, ids,
        box_size=1000.,
        num_part_1d=512,
        Omega_m=0.3175,
        redshift=127.,
        dtype='f8',
):
    """
    Writes minimal Gadget4 HDF5 snapshot files for dark matter only simulations

    Parameters
    ----------
    filename_base : str
        output is written to filename_base.hdf5 for single file output or
        filename_base.0.hdf5, filename_base.1.hdf5, etc., for multiple files
    num_files : int
        number of output files
    """
    rho_crit = 2.77536627e11

    num_part_3d = num_part_1d ** 3
    num_part_total = np.zeros(6, dtype='u4')
    num_part_total[1] = num_part_3d
    num_part_per_file = num_part_3d // num_files

    masses = np.zeros(6, dtype='f8')
    masses[num_part_total != 0] = (
        rho_crit * Omega_m * box_size ** 3
        / num_part_total[num_part_total != 0]
        * 1e-10)

    header = {
        'NumPart_Total': num_part_total,
        'MassTable': masses,
        'Time': 1. / (1. + redshift),
        'Redshift': redshift,
        'BoxSize': box_size,
        'NumFilesPerSnapshot': num_files,
        'Flag_DoublePrecision': 1 if dtype == 'f8' else 0,
    }

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
        if i < num_part_3d % num_files:
            num_part_this_file[1] += 1

        with File(filename, 'w') as f:
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
