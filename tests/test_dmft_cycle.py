#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains unit tests for dmft_cycle
"""

import numpy as np
from dmft_cycle import _mix_chemical_potential, _determine_afm_mapping
from check_equality import are_iterables_equal

def test_mix_chemical_potential():
    general_parameters = {'mu_mix_const': 0, 'mu_mix_per_occupation_offset': 1}
    density_tot = 16
    density_required = 16
    previous_mu = 0.0
    predicted_mu = 1.0

    new_mu = _mix_chemical_potential(general_parameters, density_tot, density_required,
                                     previous_mu, predicted_mu)
    assert np.isclose(new_mu, 0)

    general_parameters = {'mu_mix_const': 0, 'mu_mix_per_occupation_offset': 1}
    density_tot = 15.5
    density_required = 16
    previous_mu = 0.0
    predicted_mu = 1.0

    new_mu = _mix_chemical_potential(general_parameters, density_tot, density_required,
                                     previous_mu, predicted_mu)
    assert np.isclose(new_mu, .5)

    general_parameters = {'mu_mix_const': 1, 'mu_mix_per_occupation_offset': 0}
    density_tot = 12.34
    density_required = 16
    previous_mu = 0.0
    predicted_mu = 1.0

    new_mu = _mix_chemical_potential(general_parameters, density_tot, density_required,
                                     previous_mu, predicted_mu)
    assert np.isclose(new_mu, 1.)

    general_parameters = {'mu_mix_const': .3, 'mu_mix_per_occupation_offset': 1.}
    density_tot = 15.8
    density_required = 16
    previous_mu = 0.0
    predicted_mu = 1.0

    new_mu = _mix_chemical_potential(general_parameters, density_tot, density_required,
                                     previous_mu, predicted_mu)
    assert np.isclose(new_mu, .5)


def test_determine_afm_mapping():
    general_parameters = {'magmom': [+1, -1, +1, -1], 'afm_order': True}
    archive = {'DMFT_input': {}}
    n_inequiv_shells = 4

    expected_general_parameters = general_parameters.copy()
    #                                            copy, source, switch
    expected_general_parameters['afm_mapping'] = [[False, 0, False], [True, 0, True],
                                                  [True, 0, False], [True, 0, True]]

    general_parameters = _determine_afm_mapping(general_parameters, archive, n_inequiv_shells)

    assert are_iterables_equal(general_parameters, expected_general_parameters)

    general_parameters = {'magmom': [+1, -1, +2, +2], 'afm_order': True}
    archive = {'DMFT_input': {}}
    n_inequiv_shells = 4

    expected_general_parameters = general_parameters.copy()
    #                                            copy, source, switch
    expected_general_parameters['afm_mapping'] = [[False, 0, False], [True, 0, True],
                                                  [False, 2, False], [True, 2, False]]

    general_parameters = _determine_afm_mapping(general_parameters, archive, n_inequiv_shells)

    assert are_iterables_equal(general_parameters, expected_general_parameters)

    # Reading in the afm_mapping from the archive
    general_parameters = {'magmom': [+1, -1, +2], 'afm_order': True}
    archive = {'DMFT_input': {'afm_mapping': [[False, 0, False], [False, 1, False],
                                              [False, 2, False]]}}
    n_inequiv_shells = 3

    expected_general_parameters = general_parameters.copy()
    #                                            copy, source, switch
    expected_general_parameters['afm_mapping'] = [[False, 0, False], [False, 1, False],
                                                  [False, 2, False]]

    general_parameters = _determine_afm_mapping(general_parameters, archive, n_inequiv_shells)

    assert are_iterables_equal(general_parameters, expected_general_parameters)

    general_parameters = {'magmom': [+1, -1, +2, +2], 'afm_order': True}
    archive = {'DMFT_input': {}}
    n_inequiv_shells = 3

    expected_general_parameters = general_parameters.copy()
    #                                            copy, source, switch
    expected_general_parameters['afm_order'] = False

    general_parameters = _determine_afm_mapping(general_parameters, archive, n_inequiv_shells)

    assert are_iterables_equal(general_parameters, expected_general_parameters)
