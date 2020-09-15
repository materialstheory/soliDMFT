#!/bin/python

"""
Reads the _bands.dat and the _bands.kpt file (as Wannier90 writes them).
The _bands.dat has the band energies in the second column and different bands
are separated by an empty line. The _bands.kpt has the number of k points
in the first line and then the list of k points in 3D direct coordinates.

This only works for k independent projectors as from a TB model or from Wannier90.

Writes all the information back into the h5 archive in the group 'dft_bands_input',
which is needed for plotting DMFT bands with SumkDFTTools spaghettis.

Written by Maximilian Merkel, 2020
"""

import sys
import numpy as np
from h5 import HDFArchive


def _read_bands(seedname):
    """ Reads the _bands.dat and the _bands.kpt file. """

    print('Reading {0}_band.dat and {0}_band.kpt'.format(seedname))

    kpoints = np.loadtxt('{}_band.kpt'.format(seedname), skiprows=1, usecols=(0, 1, 2))

    band_energies = [[]]
    with open('{}_band.dat'.format(seedname)) as file:
        for line in file:
            if line.strip() == '':
                band_energies.append([])
            else:
                data_line = [s for s in line.split() if s]
                band_energies[-1].append(float(data_line[1]))

    # Deals with empty lines at the end
    while band_energies[-1] == []:
        del band_energies[-1]

    band_energies = np.array([np.diag(e) for e in np.transpose(band_energies)], dtype=complex)
    band_energies = band_energies.reshape((kpoints.shape[0], 1,
                                           band_energies.shape[1], band_energies.shape[1]))

    return kpoints, band_energies


def _read_h5_dft_input_proj_mat(archive_name):
    """
    Reads the projection matrix from the h5. In the following,
    it is assumed to be k independent.
    """
    with HDFArchive(archive_name, 'r') as archive:
        return archive['dft_input/proj_mat']


def _write_dft_bands_input_to_h5(archive_name, data):
    """Writes all the information back to the h5 archive. data is a dict. """
    with HDFArchive(archive_name, 'a') as archive:
        if 'dft_bands_input' in archive:
            del archive['dft_bands_input']
        archive.create_group('dft_bands_input')
        for key in data:
            archive['dft_bands_input'][key] = data[key]
    print('Written results to {}'.format(archive_name))


def main(seedname, filename_archive=None):
    """
    Executes the program on the band data from the files <seedname>_bands.dat and
    <seedname>_bands.kpt. If no seedname_archive is specified, <seedname>.h5 is used.
    """

    if filename_archive is None:
        filename_archive = seedname + '.h5'
        print('Using the archive "{}"'.format(filename_archive))

    kpoints, band_energies = _read_bands(seedname)
    dft_proj_mat = _read_h5_dft_input_proj_mat(filename_archive)

    data = {'n_k': kpoints.shape[0],
            'n_orbitals': np.ones((kpoints.shape[0], 1), dtype=int) * band_energies.shape[2], # The 1 in here only works for SO == 0
            'proj_mat': np.broadcast_to(dft_proj_mat[0],
                                        (kpoints.shape[0], ) + dft_proj_mat.shape[1:]),
            'hopping': band_energies,
            # Quantities are not used for unprojected spaghetti
            'n_parproj': 'none',
            'proj_mat_all': 'none',
            # Quantity that SumkDFTTools does not need but that is nice for plots
            'kpoints': kpoints}

    _write_dft_bands_input_to_h5(filename_archive, data)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Please give a seedname (and optionally an archive to write to). Exiting.')
        sys.exit(2)
