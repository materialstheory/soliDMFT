#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains unit tests for read_config.py
"""


from configparser import ConfigParser

from read_config import (_config_find_default_section_entries, _config_apply_sections_legacy_name_mapping,
                         _config_add_empty_sections, _config_remove_unused_sections)
from read_config import (_convert_parameters, _find_nonexistent_parameters,
                         _apply_default_values, _check_if_parameters_used,
                         _checks_validity_criterion)

from check_equality import are_iterables_equal

CONFIG_FILE_1 = u'''
[general]
seedname = fancy_system
jobname = out_DMFT_fancy
enforce_off_diag = True
block_threshold= 0.001
set_rot = none

prec_mu = -0.001

h_int_type = 1
U = 2.0
J = 2.0

n_iter_dmft = 4

dc_type = 0
dc = True
dc_dmft = False

calc_energies = True
sigma_mix = 0.7

h5_save_freq = 5

n_iter_dmft_per = 5

[advanced_parameters]
nonexistent = 2

[weird_parameters]
test = 2
system_size = 1e23
'''

def test_config_file_1():
    config = ConfigParser()
    config.read_string(CONFIG_FILE_1)

    # Checks if default section is empty
    config_default_entries = _config_find_default_section_entries(config)
    assert config_default_entries == []

    # Applies mapping of legacy names for sections and prints warnings
    config, duplicate_sections, renamed_sections = _config_apply_sections_legacy_name_mapping(config)
    assert duplicate_sections == []
    assert renamed_sections == [('advanced', 'advanced_parameters')]

    # Adds empty sections if they don't exist
    config = _config_add_empty_sections(config)

    # Removes unused sections and prints a warning
    config, unused_sections = _config_remove_unused_sections(config)
    assert unused_sections == ['weird_parameters']

    parameters = _convert_parameters(config)

    nonexistent_parameters = _find_nonexistent_parameters(config)
    assert nonexistent_parameters == {'dft': [], 'general': [], 'advanced': ['nonexistent'], 'solver': []}

    parameters, default_values_used = _apply_default_values(parameters)

    parameters, unnecessary_parameters, missing_required_parameters = _check_if_parameters_used(parameters, default_values_used)
    assert unnecessary_parameters == {'dft': [], 'general': ['n_iter_dmft_per'], 'advanced': [], 'solver': []}
    assert missing_required_parameters == {'dft': [], 'general': ['beta'], 'advanced': [],
                                           'solver': ['n_warmup_cycles', 'length_cycle', 'n_cycles_tot']}

    invalid_parameters = _checks_validity_criterion(parameters)
    assert invalid_parameters == {'dft': [], 'general': ['prec_mu'], 'advanced': [], 'solver': []}

    assert are_iterables_equal(parameters, {'dft': {},
                                            'general': {'magnetic': False, 'fixed_mu_value': 'none',
                                                        'mu_update_freq': 1,
                                                        'measure_chi_SzSz': False, 'block_threshold': 0.001,
                                                        'set_rot': u'none', 'prec_mu': -0.001,
                                                        'dft_mu': u'none',
                                                        'mu_mix_const': 1., 'mu_mix_per_occupation_offset': 0.,
                                                        'oneshot_postproc_gamma_file': False,
                                                        'csc': False, 'enforce_off_diag': True,
                                                        'store_dft_eigenvals': False, 'dc_dmft': False,
                                                        'occ_conv_crit': -1, 'seedname': [u'fancy_system'],
                                                        'J': [2.0], 'h5_save_freq': 5,
                                                        'dc': True, 'jobname': [u'out_DMFT_fancy'],
                                                        'n_iter_dmft': 4, 'U': [2.0],
                                                        'measure_chi_insertions': 100, 'h_field': 0.0,
                                                        'calc_energies': True, 'sigma_mix': 0.7,
                                                        'dc_type': 0, 'load_sigma': False,
                                                        'noise_level_initial_sigma': 0.,
                                                        'h_int_type': 1, 'spin_names': ['up', 'down']},
                                            'advanced': {'dc_fixed_value': 'none', 'dc_fixed_occ': 'none',
                                                         'dc_nominal': False,
                                                         'dc_factor': 'none', 'dc_J': [2.0], 'dc_U': [2.0]},
                                            'solver': {'move_double': True, 'measure_G_l': False,
                                                       'move_shift': False, 'store_solver': False,
                                                       'measure_G_tau': True, 'measure_pert_order': False,
                                                       'measure_density_matrix': False, 'perform_tail_fit': False,
                                                       'legendre_fit': False}}
                              )



CONFIG_FILE_2 = u'''
[general]
seedname = orbital_model
jobname = out_60M_afm
enforce_off_diag = True
block_threshold= 0.001
set_rot = none

prec_mu = 0.001

h_int_type = 1
U = 5.5
J = 1.0

beta = 40

n_iter_dmft = 6

path_to_sigma = orbital_2site_model_Sigma_eg_swapped.h5

dc_type = 0
dc = True
dc_dmft = False

magnetic = True
magmom = 1, -1
afm_order = True

calc_energies = False
sigma_mix = 0.6

h5_save_freq = 2

[solver_parameters]
length_cycle = 140
n_warmup_cycles = 10000
n_cycles_tot = 60e+6
imag_threshold = 1e-5
measure_G_l = True
n_LegCoeff = 35
measure_g_tau = False

measure_density_matrix = False

perform_tail_fit = True
'''

def test_config_file_2():
    config = ConfigParser()
    config.read_string(CONFIG_FILE_2)

    # Checks if default section is empty
    config_default_entries = _config_find_default_section_entries(config)
    assert config_default_entries == []

    # Applies mapping of legacy names for sections and prints warnings
    config, duplicate_sections, renamed_sections = _config_apply_sections_legacy_name_mapping(config)
    assert duplicate_sections == []
    assert renamed_sections == [('solver', 'solver_parameters')]

    # Adds empty sections if they don't exist
    config = _config_add_empty_sections(config)

    # Removes unused sections and prints a warning
    config, unused_sections = _config_remove_unused_sections(config)
    assert unused_sections == []

    parameters = _convert_parameters(config)

    nonexistent_parameters = _find_nonexistent_parameters(config)
    assert nonexistent_parameters == {'dft': [], 'general': [], 'advanced': [], 'solver': []}

    parameters, default_values_used = _apply_default_values(parameters)

    parameters, unnecessary_parameters, missing_required_parameters = _check_if_parameters_used(parameters, default_values_used)
    assert unnecessary_parameters == {'dft': [], 'general': ['path_to_sigma'],
                                      'advanced': [], 'solver': ['perform_tail_fit']}
    assert missing_required_parameters == {'dft': [], 'general': [], 'advanced': [], 'solver': []}


    invalid_parameters = _checks_validity_criterion(parameters)
    assert invalid_parameters == {'dft': [], 'general': [], 'advanced': [], 'solver': []}

    print(parameters)
    assert are_iterables_equal(parameters, {'dft': {},
                                            'general': {'afm_order': True, 'magnetic': True,
                                                        'store_dft_eigenvals': False, 'measure_chi_SzSz': False,
                                                        'block_threshold': 0.001, 'set_rot': u'none',
                                                        'prec_mu': 0.001, 'dft_mu': u'none',
                                                        'mu_mix_const': 1., 'mu_mix_per_occupation_offset': 0.,
                                                        'oneshot_postproc_gamma_file': False, 'csc': False,
                                                        'enforce_off_diag': True, 'fixed_mu_value': 'none',
                                                        'mu_update_freq': 1,
                                                        'seedname': [u'orbital_model'], 'dc_dmft': False,
                                                        'occ_conv_crit': -1, 'J': [1.0], 'h5_save_freq': 2,
                                                        'dc': True, 'jobname': [u'out_60M_afm'],
                                                        'beta': 40.0, 'U': [5.5],
                                                        'measure_chi_insertions': 100, 'h_field': 0.0,
                                                        'calc_energies': False, 'sigma_mix': 0.6,
                                                        'magmom': [1.0, -1.0], 'dc_type': 0,
                                                        'load_sigma': False,
                                                        'noise_level_initial_sigma': 0., 'n_iter_dmft': 6,
                                                        'h_int_type': 1, 'spin_names': ['up', 'down']},
                                            'advanced': {'dc_fixed_value': 'none', 'dc_fixed_occ': 'none',
                                                         'dc_nominal': False,
                                                         'dc_factor': 'none', 'dc_J': [1.0], 'dc_U': [5.5]},
                                            'solver': {'imag_threshold': 1e-05, 'n_warmup_cycles': 10000,
                                                       'length_cycle': 140, 'measure_G_l': True,
                                                       'n_cycles_tot': 60000000, 'store_solver': False,
                                                       'move_double': True, 'measure_pert_order': False,
                                                       'n_LegCoeff': 35, 'move_shift': False,
                                                       'measure_G_tau': False, 'measure_density_matrix': False}}

                             )
