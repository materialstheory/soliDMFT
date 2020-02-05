#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:35:31 2020

@author: mmerkel
"""

import sys
import numpy as np

from pytriqs.gf import GfReFreq, BlockGf
from pytriqs.archive import HDFArchive
from triqs_dft_tools.sumk_dft_tools import SumkDFTTools


def _read_h5(external_path, iteration=None):
    """Reads Sigma(omega), the DC, mu and the kpoints from the h5 """

    h5_internal_path = 'DMFT_results/' + ('last_iter' if iteration is None else 'it_{}'.format(iteration))

    with HDFArchive(external_path, 'r') as archive:
        impurity_paths = filter(lambda key: 'Sigma_w_' in key, archive[h5_internal_path].keys())
        # Sorts impurity paths by their indices, not sure if necessary
        impurity_indices = [int(s[s.rfind('_')+1:]) for s in impurity_paths]
        impurity_paths = [impurity_paths[i] for i in np.argsort(impurity_indices)]

        sigma_w = [archive[h5_internal_path][p] for p in impurity_paths]

        dc_potential = archive[h5_internal_path]['DC_pot']
        dc_energy = archive[h5_internal_path]['DC_energ']
        chemical_potential = archive[h5_internal_path]['chemical_potential']

    return sigma_w, dc_potential, dc_energy, chemical_potential


def _initialize_sum_k_tools(external_path, chemical_potential, dc_potential, dc_energy, sigma_w):
    """ Creates the SumKDFTTools objects that will calculate the spectral properties. """
    sum_k = SumkDFTTools(hdf_file=external_path, use_dft_blocks=True)
    sum_k.set_mu(chemical_potential)
    sum_k.set_dc(dc_potential, dc_energy)
    sum_k.put_Sigma(sigma_w)
    return sum_k


def _get_diagonal_sigma(sigma_w):
    """ Turns a 1 block, n x n GF into a n block, 1x1 GF"""

    # Returns unaltered Sigma if Sigma is already a n block, 1x1 GF
    is_already_diagonal = True
    for i in range(len(sigma_w)):
        for key, block in sigma_w[i]:
            print(block.data.shape)
            if block.data.shape[1] > 1:
                is_already_diagonal = False
                break
        if not is_already_diagonal:
            break

    if is_already_diagonal:
        return sigma_w

    # Makes Sigma diagonal
    new_sigma_w = []
    for i in range(len(sigma_w)):
        new_keys = []
        diag_elems = []
        for key, block in sigma_w[i]:
            if int(key[key.rfind('_')+1:]) != 0:
                raise ValueError('Can only convert BlockGf\'s with blocks \'up_0\' and \'down_0\'.')

            number_blocks = block.data.shape[1]
            new_keys += [key.replace('0', str(j)) for j in range(number_blocks)]
            diag_elems += [GfReFreq(mesh=sigma_w[i].mesh, data=block.data[:, j:j+1, j:j+1])
                           for j in range(number_blocks)]
        new_sigma_w.append(BlockGf(name_list=new_keys, block_list=diag_elems))

    return new_sigma_w


def _write_dos_and_spaghetti_to_h5(mesh, dos, dos_proj, dos_proj_orb, spaghetti,
                                   external_path, iteration=None):
    """ Writes different spectral functions and the spaghetti to the h5 archive. """

    h5_internal_path = 'DMFT_results/' + ('last_iter' if iteration is None else 'it_{}'.format(iteration))

    with HDFArchive(external_path, 'a') as archive:
        dos['mesh'] = mesh
        archive[h5_internal_path]['Alatt_w_from_Sigma_w'] = dos
        for i, (res_total, res_per_orb) in enumerate(zip(dos_proj, dos_proj_orb)):
            archive[h5_internal_path]['Aimp_w_{}_from_Sigma_w'.format(i)] = {'total': res_total,
                                                                             'per_orb': res_per_orb,
                                                                             'mesh': mesh}
        spaghetti['mesh'] = mesh
        archive[h5_internal_path]['Alatt_k_w_from_Sigma_w'] = spaghetti


def main(external_path, iteration=None):
    """
    Theoretically, spectral functions not needed but good as comparison.

    Parameters
    ----------
    external_path: string, path of the h5 archive
    iteration: int/string, optional, iteration to read from and write to

    Returns
    -------
    numpy array, omega mesh for all spectral functions
    dict with 'up' and 'down' with the lattice spectral function
    list of dict, per impurity: A_imp(omega)
    list of dict, per impurity: A_imp(omega), orbital resolved
    dict with 'up' and 'down' with the k resolved lattice spectral function
    """

    sigma_w, dc_potential, dc_energy, chemical_potential = _read_h5(external_path, iteration)
    sigma_w = _get_diagonal_sigma(sigma_w)
    sum_k = _initialize_sum_k_tools(external_path, chemical_potential,
                                    dc_potential, dc_energy, sigma_w)
    mesh = np.array([x.real for x in sigma_w[0].mesh])

    alatt_w, aimp_w, aimp_w_per_orb = sum_k.dos_wannier_basis(save_to_file=False)
    print('Calculated the spectral functions. Starting with spaghetti now.')

    alatt_k_w = sum_k.spaghettis(save_to_file=False)
    _write_dos_and_spaghetti_to_h5(mesh, alatt_w, aimp_w, aimp_w_per_orb, alatt_k_w,
                                   external_path, iteration)

    print('Calculated spaghetti and wrote all results to file.')
    return mesh, alatt_w, aimp_w, aimp_w_per_orb, alatt_k_w


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print('Please give the h5 name (and optionally the iteration). Exiting.')
        sys.exit(2)
