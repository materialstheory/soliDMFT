#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains unit tests for observables.py
"""

from triqs.gf import BlockGf, GfImFreq, GfImTime, SemiCircular
from triqs.gf.descriptors import Fourier

from observables import add_dft_values_as_zeroth_iteration, add_dmft_observables, _generate_header
from helper import are_iterables_equal, Dummy


def _set_up_observables(n_inequiv_shells):
    observables = {}
    observables['iteration'] = []
    observables['mu'] = []
    observables['E_tot'] = []
    observables['E_bandcorr'] = []
    observables['E_int'] = [[] for _ in range(n_inequiv_shells)]
    observables['E_corr_en'] = []
    observables['E_dft'] = []
    observables['E_DC'] = [[] for _ in range(n_inequiv_shells)]
    observables['orb_gb2'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
    observables['imp_gb2'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
    observables['orb_occ'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
    observables['imp_occ'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
    observables['orb_Z'] =  [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]

    return observables


# ---------- add_dft_values_as_zeroth_iteration ----------

def test_add_dft_values_one_impurity_one_band():
    sum_k = Dummy()
    sum_k.n_inequiv_shells = 1
    sum_k.dc_energ = [0.0]
    sum_k.inequiv_to_corr = [0]
    sum_k.gf_struct_solver = [{'up_0': None, 'down_0': None}]
    sum_k.spin_block_names = [['up', 'down'], ['ud']]
    sum_k.SO = 0

    general_parameters = {}
    general_parameters['calc_energies'] = False
    general_parameters['csc'] = False
    general_parameters['dc'] = False
    general_parameters['beta'] = 40.

    gf_up = GfImFreq(indices=[0], beta=general_parameters['beta'])
    gf_down = gf_up.copy()
    gf_up << SemiCircular(2, 0)
    gf_down << SemiCircular(1, 0)
    G_loc_all_dft = [BlockGf(name_list=('up_0', 'down_0'), block_list=(gf_up, gf_down), make_copies=True)]

    density_mat_dft = [G_loc_all_dft[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]
    dft_mu = 12.
    shell_multiplicity = [4]

    observables = _set_up_observables(sum_k.n_inequiv_shells)

    observables = add_dft_values_as_zeroth_iteration(observables, general_parameters, dft_mu, sum_k,
                                                     G_loc_all_dft, density_mat_dft, shell_multiplicity)

    expected_observables = {'E_bandcorr': ['none'], 'E_tot': ['none'], 'E_dft': ['none'],
                            'E_DC': [['none']], 'E_int': [['none']],
                            'orb_occ': [{'down': [[0.5]], 'up': [[0.5]]}],
                            'iteration': [0], 'E_corr_en': ['none'], 'mu': [12.0],
                            'orb_gb2': [{'down': [[-0.0498]], 'up': [[-0.0250]]}],
                            'imp_gb2': [{'down': [-0.0498], 'up': [-0.0250]}],
                            'imp_occ': [{'down': [0.5], 'up': [0.5]}],
                            'orb_Z': [{'down': [[1.0]], 'up': [[1.0]]}]}

    assert are_iterables_equal(observables, expected_observables)


def test_add_dft_values_two_impurites_two_bands():
    sum_k = Dummy()
    sum_k.n_inequiv_shells = 2
    # mapping corr to inequiv = [0, 0, 1, 0]
    sum_k.dc_energ = [0.1, 0.1, 5.0, 0.1]
    sum_k.inequiv_to_corr = [0, 2]
    sum_k.gf_struct_solver = [{'up_0': None, 'down_0': None, 'up_1': None, 'down_1': None},
                              {'up_0': None, 'down_0': None}]
    sum_k.spin_block_names = [['up', 'down'], ['ud']]
    sum_k.SO = 0

    general_parameters = {}
    general_parameters['calc_energies'] = True
    general_parameters['csc'] = False
    general_parameters['dc'] = True
    general_parameters['beta'] = 40.

    gf_up_one_band = GfImFreq(indices=[0], beta=general_parameters['beta'])
    gf_down_one_band = gf_up_one_band.copy()
    gf_up_one_band << SemiCircular(2, 1)
    gf_down_one_band << SemiCircular(1, 2)

    gf_up_two_bands = GfImFreq(indices=[0, 1], beta=general_parameters['beta'])
    gf_down_two_bands = gf_up_two_bands.copy()
    gf_up_two_bands << SemiCircular(2, 0)
    gf_down_two_bands << SemiCircular(1, 0)

    G_loc_all_dft = [BlockGf(name_list=('up_0', 'down_1', 'down_0', 'up_1'),
                             block_list=(gf_up_one_band, gf_down_one_band, gf_down_one_band, gf_up_one_band),
                             make_copies=True),
                     BlockGf(name_list=('up_0', 'down_0'), block_list=(gf_up_two_bands, gf_down_two_bands), make_copies=True)]

    density_mat_dft = [G_loc_all_dft[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]
    dft_mu = 2.
    shell_multiplicity = [3, 1]

    observables = _set_up_observables(sum_k.n_inequiv_shells)

    observables = add_dft_values_as_zeroth_iteration(observables, general_parameters, dft_mu, sum_k,
                                                     G_loc_all_dft, density_mat_dft, shell_multiplicity)
    expected_observables = {'E_bandcorr': [0.0], 'E_tot': [-5.3], 'E_dft': [0.0],
                            'E_DC': [[0.3], [5.0]], 'E_int': [[0.0], [0.0]],
                            'orb_occ': [{'down': [[1., 1.]], 'up': [[0.8044, 0.8044]]}, {'down': [[0.5, 0.5]], 'up': [[0.5, 0.5]]}],
                            'iteration': [0], 'E_corr_en': [0.0], 'mu': [2.0],
                            'orb_gb2': [{'down': [[0, 0]], 'up': [[-0.0216, -0.0216]]}, {'down': [[-0.0498, -0.0498]], 'up': [[-0.0250, -0.0250]]}],
                            'imp_gb2': [{'down': [0], 'up': [-0.0432]}, {'down': [-0.0996], 'up': [-0.0500]}],
                            'imp_occ': [{'down': [2.], 'up': [1.6088]}, {'down': [1.], 'up': [1.]}],
                            'orb_Z': [{'down': [[1., 1.]], 'up': [[1., 1.]]}, {'down': [[1., 1.]], 'up': [[1., 1.]]}],
                           }

    assert are_iterables_equal(observables, expected_observables)


# ---------- add_dmft_observables ----------

def test_add_dmft_observables_one_impurity_one_band():
    sum_k = Dummy()
    sum_k.n_inequiv_shells = 1
    sum_k.dc_energ = [0.0]
    sum_k.inequiv_to_corr = [0]
    sum_k.gf_struct_solver = [{'up_0': None, 'down_0': None}]
    sum_k.spin_block_names = [['up', 'down'], ['ud']]
    sum_k.SO = 0

    general_parameters = {}
    general_parameters['calc_energies'] = False
    general_parameters['csc'] = False
    general_parameters['dc'] = False
    general_parameters['beta'] = 40.

    solver_parameters = {'measure_density_matrix': False}

    observables = _set_up_observables(sum_k.n_inequiv_shells)

    it = 1
    h_int = None
    previous_mu = 1.56
    shell_multiplicity = [4]
    E_bandcorr = 10.43

    gf_up = GfImFreq(indices=[0], beta=general_parameters['beta'])
    gf_down = gf_up.copy()
    gf_up << SemiCircular(2, 0)
    gf_down << SemiCircular(1, 0)
    gf_iw = [BlockGf(name_list=('up_0', 'down_0'), block_list=(gf_up, gf_down), make_copies=True)]

    gf_imtime = GfImTime(indices=[0], beta=general_parameters['beta'])
    gf_tau = [BlockGf(name_list=('up_0', 'down_0'), block_list=(gf_imtime, gf_imtime), make_copies=True)]
    gf_tau[0] << Fourier(gf_iw[0])

    solvers = [Dummy()]
    solvers[0].G_tau = gf_tau[0]
    # misusing here G_iw to get some comparable dummy values for Z
    solvers[0].Sigma_iw = gf_iw[0]

    density_mat = [gf_iw[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]


    observables = add_dmft_observables(observables, general_parameters, solver_parameters, it, solvers, h_int,
                                       previous_mu, sum_k, density_mat, shell_multiplicity, E_bandcorr)

    expected_observables = {'iteration': [1], 'mu': [1.56], 'E_tot': [10.43], 'E_bandcorr': [10.43],
                            'E_int': [[]], 'E_corr_en': [0.0], 'E_dft': [0.0], 'E_DC': [[0.0]],
                            'orb_gb2': [{'up': [[-0.0250]], 'down': [[-0.0498]]}], 'imp_gb2': [{'up': [-0.0250], 'down': [-0.0498]}],
                            'orb_occ': [{'up': [[0.5]], 'down': [[0.5]]}],
                            'imp_occ': [{'up': [0.5], 'down': [0.5]}],
                            'orb_Z':   [{'up': [[0.0755]], 'down': [[0.0407]]}]
                            }

    assert are_iterables_equal(observables, expected_observables)


# ---------- _generate_header ----------

def test_generate_header():
    general_parameters = {}
    general_parameters['magnetic'] = False
    general_parameters['calc_energies'] = False
    general_parameters['csc'] = False
    sum_k = Dummy()
    sum_k.n_inequiv_shells = 1
    sum_k.inequiv_to_corr = [0]
    sum_k.corr_shells = [{'dim': 3}]
    sum_k.SO = 0

    headers = _generate_header(general_parameters, sum_k)
    expected_headers = {'observables_imp0.dat':
                            ' it |         mu |                G(beta/2) per orbital |'
                            + '                 orbital occs up+down |      impurity occ'}
    assert headers == expected_headers


    general_parameters = {}
    general_parameters['magnetic'] = False
    general_parameters['calc_energies'] = True
    general_parameters['csc'] = False
    sum_k = Dummy()
    sum_k.n_inequiv_shells = 2
    sum_k.inequiv_to_corr = [0, 2]
    sum_k.corr_shells = [{'dim': 3}, {'dim': 3}, {'dim': 1}, {'dim': 1}]
    sum_k.SO = 0

    headers = _generate_header(general_parameters, sum_k)
    expected_headers = {'observables_imp0.dat':
                            ' it |         mu |                G(beta/2) per orbital |'
                            + '                 orbital occs up+down |      impurity occ |'
                            + '      E_tot |      E_DFT   E_bandcorr    E_int_imp         E_DC',
                        'observables_imp1.dat':
                            ' it |         mu | G(beta/2) per orbital |'
                            + '  orbital occs up+down |      impurity occ |'
                            + '      E_tot |      E_DFT   E_bandcorr    E_int_imp         E_DC'}
    assert headers == expected_headers

    general_parameters = {}
    general_parameters['magnetic'] = True
    general_parameters['calc_energies'] = True
    general_parameters['csc'] = False
    sum_k = Dummy()
    sum_k.n_inequiv_shells = 2
    sum_k.inequiv_to_corr = [0, 1]
    sum_k.corr_shells = [{'dim': 2}, {'dim': 1}]
    sum_k.SO = 0

    headers = _generate_header(general_parameters, sum_k)
    expected_headers = {'observables_imp0_up.dat':
                            ' it |         mu |   G(beta/2) per orbital |'
                            + '         orbital occs up |   impurity occ up |'
                            + '      E_tot |      E_DFT   E_bandcorr    E_int_imp         E_DC',
                        'observables_imp0_down.dat':
                            ' it |         mu |   G(beta/2) per orbital |'
                            + '       orbital occs down | impurity occ down |'
                            + '      E_tot |      E_DFT   E_bandcorr    E_int_imp         E_DC',
                        'observables_imp1_up.dat':
                            ' it |         mu | G(beta/2) per orbital |'
                            + '       orbital occs up |   impurity occ up |'
                            + '      E_tot |      E_DFT   E_bandcorr    E_int_imp         E_DC',
                        'observables_imp1_down.dat':
                            ' it |         mu | G(beta/2) per orbital |'
                            + '     orbital occs down | impurity occ down |'
                            + '      E_tot |      E_DFT   E_bandcorr    E_int_imp         E_DC'}
    assert headers == expected_headers
