"""
Defines the dmft_cycle which works for one-shot and csc equally
"""

# the future numpy (>1.15) is not fully compatible with triqs 2.0 atm
# suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system
import numpy as np
import os
from copy import deepcopy

# triqs
import pytriqs.utility.mpi as mpi
from pytriqs.operators.util.U_matrix import (U_matrix, U_matrix_kanamori, reduce_4index_to_2index,
                                             U_J_to_radial_integrals, transform_U_matrix)
from pytriqs.operators.util.hamiltonians import h_int_kanamori, h_int_density, h_int_slater
from pytriqs.archive import HDFArchive
from triqs_cthyb.solver import Solver
from triqs_dft_tools.sumk_dft import SumkDFT
from pytriqs.gf import GfImTime, GfImFreq, GfLegendre, BlockGf, make_hermitian
from pytriqs.gf.tools import inverse
from pytriqs.gf.descriptors import Fourier


# own modules
from observables import (calc_dft_kin_en, add_dmft_observables, calc_bandcorr_man, write_obs,
                         add_dft_values_as_zeroth_iteration, write_header_to_file)
import toolset

def _calculate_double_counting(sum_k, density_matrix, general_parameters, advanced_parameters):
    """
    Calculates the double counting, including all manipulations from advanced_parameters.

    Parameters
    ----------
    sum_k : SumkDFT object
    density_matrix : list of gf_struct_solver like
        List of density matrices for all inequivalent shells
    general_parameters : dict
        general parameters as a dict
    advanced_parameters : dict
        advanced parameters as a dict

    __Returns:__
    sum_k : SumKDFT object
        The SumKDFT object containing the updated double counting
    """

    mpi.report('\n*** DC determination ***')

    # copy the density matrix to not change it
    density_matrix_DC = deepcopy(density_matrix)

    # Sets the DC and exits the function if advanced_parameters['dc_fixed_value'] is specified
    if advanced_parameters['dc_fixed_value'] != 'none':
        for icrsh in range(sum_k.n_inequiv_shells):
            sum_k.calc_dc(density_matrix_DC[icrsh], orb=icrsh,
                          use_dc_value=advanced_parameters['dc_fixed_value'])
        return sum_k


    if advanced_parameters['dc_fixed_occ'] != 'none':
        mpi.report('Fixing occupation for DC potential to provided value')

        assert sum_k.n_inequiv_shells == len(advanced_parameters['dc_fixed_occ']), "give exactly one occupation per correlated shell"
        for icrsh in range(sum_k.n_inequiv_shells):
            mpi.report('fixing occupation for impurity '+str(icrsh)+' to n='+str(advanced_parameters['dc_fixed_occ'][icrsh]))
            n_orb = sum_k.corr_shells[icrsh]['dim']
            # we need to handover a matrix to calc_dc so calc occ per orb per spin channel
            orb_occ = advanced_parameters['dc_fixed_occ'][icrsh]/(n_orb*2)
            # setting occ of each diag orb element to calc value
            for block, inner in density_matrix_DC[icrsh].items():
                np.fill_diagonal(inner, orb_occ+0.0j)

    # The regular way: calculates the DC based on U, J and the dc_type
    for icrsh in range(sum_k.n_inequiv_shells):
        if general_parameters['dc_type'] == 3:
            # this is FLL for eg orbitals only as done in Seth PRB 96 205139 2017 eq 10
            # this setting for U and J is reasonable as it is in the spirit of F0 and Javg
            # for the 5 orb case
            mpi.report('Doing FLL DC for eg orbitals only with Uavg=U-J and Javg=2*J')
            Uavg = advanced_parameters['dc_U'][icrsh] - advanced_parameters['dc_J'][icrsh]
            Javg = 2*advanced_parameters['dc_J'][icrsh]
            sum_k.calc_dc(density_matrix_DC[icrsh], U_interact=Uavg, J_hund=Javg,
                          orb=icrsh, use_dc_formula=0)
        else:
            sum_k.calc_dc(density_matrix_DC[icrsh], U_interact=advanced_parameters['dc_U'][icrsh],
                          J_hund=advanced_parameters['dc_J'][icrsh], orb=icrsh,
                          use_dc_formula=general_parameters['dc_type'])

    # for the fixed DC according to https://doi.org/10.1103/PhysRevB.90.075136
    # dc_imp is calculated with fixed occ but dc_energ is calculated with given n
    if advanced_parameters['dc_nominal'] == True:
        mpi.report('\ncalculating DC energy with fixed DC potential from above \n for the original density matrix doi.org/10.1103/PhysRevB.90.075136 \n aka nominal DC')
        dc_imp = deepcopy(sum_k.dc_imp)
        dc_new_en = deepcopy(sum_k.dc_energ)
        for ish in range(sum_k.n_corr_shells):
            n_DC = 0.0
            for key, value in density_matrix[sum_k.corr_to_inequiv[ish]].items():
                n_DC += np.trace(value.real)

            # calculate new DC_energ as n*V_DC
            # average over blocks in case blocks have different imp
            dc_new_en[ish] = 0.0
            for spin, dc_per_spin in dc_imp[ish].items():
                # assuming that the DC potential is the same for all orbitals
                # dc_per_spin is a list for each block containing on the diag
                # elements the DC potential for the self-energy correction
                dc_new_en[ish] += n_DC * dc_per_spin[0][0]
            dc_new_en[ish] = dc_new_en[ish] / len(dc_imp[ish])
        sum_k.set_dc(dc_imp, dc_new_en)

        # Print new DC values
        mpi.report('\nFixed occ, new DC values:')
        for icrsh, (dc_per_shell, energy_per_shell) in enumerate(zip(dc_imp, dc_new_en)):
            for spin, dc_per_spin in dc_per_shell.items():
                mpi.report('DC for shell {} and block {} = {}'.format(icrsh, spin, dc_per_spin[0][0]))
            mpi.report('DC energy for shell {} = {}'.format(icrsh, energy_per_shell))

    # Rescales DC if advanced_parameters['dc_factor'] is given
    if advanced_parameters['dc_factor'] != 'none':
        rescaled_dc_imp = [{spin: advanced_parameters['dc_factor'] * dc_per_spin
                            for spin, dc_per_spin in dc_per_shell.items()}
                          for dc_per_shell in sum_k.dc_imp]
        rescaled_dc_energy = [advanced_parameters['dc_factor'] * energy_per_shell
                              for energy_per_shell in sum_k.dc_energ]
        sum_k.set_dc(rescaled_dc_imp, rescaled_dc_energy)

        # Print new DC values
        mpi.report('\nRescaled DC, new DC values:')
        for icrsh, (dc_per_shell, energy_per_shell) in enumerate(zip(rescaled_dc_imp, rescaled_dc_energy)):
            for spin, dc_per_spin in dc_per_shell.items():
                mpi.report('DC for shell {} and block {} = {}'.format(icrsh, spin, dc_per_spin[0][0]))
            mpi.report('DC energy for shell {} = {}'.format(icrsh, energy_per_shell))

    return sum_k


def _mix_chemical_potential(general_parameters, density_tot, density_required,
                            previous_mu, predicted_mu):
    """
    Mixes the previous chemical potential and the predicted potential with linear
    mixing:
    new_mu = factor * predicted_mu + (1-factor) * previous_mu, with
    factor = mu_mix_per_occupation_offset * |density_tot - density_required| + mu_mix_const
    under the constrain of 0 <= factor <= 1.

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    density_tot : float
        total occupation of the correlated system
    density_required : float
        required density for the impurity problem
    previous_mu : float
        the chemical potential from the previous iteration
    predicted_mu : float
        the chemical potential predicted by methods like the SumkDFT dichotomy

    Returns
    -------
    new_mu : float
        the chemical potential that results from the mixing

    """
    mu_mixing = general_parameters['mu_mix_per_occupation_offset'] * abs(density_tot - density_required)
    mu_mixing += general_parameters['mu_mix_const']
    mu_mixing = max(min(mu_mixing, 1), 0)
    new_mu = mu_mixing * predicted_mu + (1-mu_mixing) * previous_mu

    mpi.report('Mixing dichotomy mu with previous iteration by factor {:.3f}'.format(mu_mixing))
    mpi.report('New chemical potential: {:.3f}'.format(new_mu))
    return new_mu


def _set_loaded_sigma(sum_k, loaded_sigma, loaded_dc_imp):
    """
    Adjusts for the Hartree shift when loading a self energy Sigma_iw from a
    previous calculation that was run with a different U, J or double counting.

    Parameters
    ----------
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    loaded_sigma : list of BlockGf (Green's function) objects
        List of Sigmas loaded from the previous calculation
    loaded_dc_imp : list of dicts
        List of dicts containing the loaded DC. Used to adjust the Hartree shift.

    Raises
    ------
    ValueError
        Raised if the block structure between the loaded and the Sumk DC_imp
        does not agree.

    Returns
    -------
    start_sigma : list of BlockGf (Green's function) objects
        List of Sigmas, loaded Sigma adjusted for the new Hartree term

    """
    # Compares loaded and new double counting
    if len(loaded_dc_imp) != len(sum_k.dc_imp):
        raise ValueError('Loaded double counting has a different number of '
                         + 'correlated shells than current calculation.')

    has_double_counting_changed = False
    for loaded_dc_shell, calc_dc_shell in zip(loaded_dc_imp, sum_k.dc_imp):
        if sorted(loaded_dc_shell.keys()) != sorted(calc_dc_shell.keys()):
            raise ValueError('Loaded double counting has a different block '
                             + 'structure than current calculation.')

        for channel in loaded_dc_shell.keys():
            if not np.allclose(loaded_dc_shell[channel], calc_dc_shell[channel],
                               atol=1e-4, rtol=0):
                has_double_counting_changed = True
                break

    # Sets initial Sigma
    start_sigma = loaded_sigma

    if not has_double_counting_changed:
        print('DC remained the same. Using loaded Sigma as initial Sigma.')
        return start_sigma

    # Uses the SumkDFT add_dc routine to correctly substract the DC shift
    sum_k.put_Sigma(start_sigma)
    calculated_dc_imp = sum_k.dc_imp
    sum_k.dc_imp = [{channel: np.array(loaded_dc_shell[channel]) - np.array(calc_dc_shell[channel])
                     for channel in loaded_dc_shell}
                    for calc_dc_shell, loaded_dc_shell in zip(sum_k.dc_imp, loaded_dc_imp)]
    start_sigma = sum_k.add_dc()
    start_sigma = toolset.sumk_sigma_to_solver_struct(sum_k, start_sigma)

    # Prints information on correction of Hartree shift
    first_block = sorted(key for key, _ in loaded_sigma[0])[0]
    print('DC changed, initial Sigma is the loaded Sigma with corrected Hartree shift:')
    print('    Sigma for imp0, block "{}", orbital 0 '.format(first_block)
          + 'shifted from {:.3f} eV '.format(loaded_sigma[0][first_block].data[0, 0, 0].real)
          + 'to {:.3f} eV'.format(start_sigma[0][first_block].data[0, 0, 0].real))

    # Cleans up
    sum_k.dc_imp = calculated_dc_imp
    [sigma_iw.zero() for sigma_iw in sum_k.Sigma_imp_iw]

    return start_sigma


def _extract_U_J_list(param_name, n_inequiv_shells, general_parameters):
    """
    Checks if param_name ('U' or 'J') are a single value or different per
    inequivalent shell. If just a single value is given, this value is
    applied to each shell.
    """

    if len(general_parameters[param_name]) == 1:
        mpi.report('Assuming {} = '.format(param_name)
                   + '{:.2f} for all correlated shells'.format(general_parameters[param_name][0]))
        general_parameters[param_name] *= n_inequiv_shells
    elif len(general_parameters[param_name]) == n_inequiv_shells:
        mpi.report('{} list for correlated shells: '.format(param_name)
                   + str(general_parameters[param_name]))
    else:
        raise IndexError('Property list {} '.format(general_parameters[param_name])
                         + 'must have length 1 or n_inequiv_shells')

    return general_parameters


def _calculate_rotation_matrix(general_parameters, sum_k, iteration_offset):
    """
    Applies rotation matrix to make the DMFT calculations easier for the solver.
    Possible are rotations diagonalizing either the local Hamiltonian or the
    density. Diagonalizing the density has not proven really helpful but
    diagonalizing the local Hamiltonian has.
    In the current implementation, this cannot be used when there is a
    non-identity rot_mat already as for example done by the Wannier90 converter.
    Note that the interaction Hamiltonian has to be rotated if it is not fully
    orbital-gauge invariant (only the Kanamori fulfills that).
    """

    # Extracts new rotation matrices from density_mat or local Hamiltonian
    if general_parameters['set_rot'] == 'hloc':
        q_diag = sum_k.eff_atomic_levels()
        chnl = 'up'
    elif general_parameters['set_rot'] == 'den':
        q_diag = sum_k.density_matrix(method='using_gf', beta=general_parameters['beta'])
        chnl = 'up'
    else:
        raise ValueError('Parameter set_rot set to wrong value.')

    rot_mat = []
    for icrsh in range(sum_k.n_corr_shells):
        ish = sum_k.corr_to_inequiv[icrsh]
        eigvec = np.linalg.eigh(np.real(q_diag[ish][chnl]))[1]
        rot_mat.append(np.array(eigvec, dtype=complex))

    sum_k.rot_mat = rot_mat
    mpi.report('Updating rotation matrices using dft {} eigenbasis to maximise sign'.format(general_parameters['set_rot']))

    # Prints matrices
    if mpi.is_master_node():
        print('\nNew rotation matrices')
        for icrsh, rot_crsh in enumerate(sum_k.rot_mat):
            n_orb = sum_k.corr_shells[icrsh]['dim']
            print('rot_mat[{:2d}] '.format(icrsh)+'real part'.center(9*n_orb)+'  '+'imaginary part'.center(9*n_orb))
            fmt = '{:9.5f}' * n_orb
            for row in rot_crsh:
                row = np.concatenate((row.real, row.imag))
                print((' '*11 + fmt + '  ' + fmt).format(*row))
        print('\n')

    return sum_k


def _construct_interaction_hamiltonian(sum_k, general_parameters):
    """
    Constructs the interaction Hamiltonian. Currently implemented are the
    Kanamori Hamiltonian (for 2 or 3 orbitals) and the density-density Hamiltonian
    (for 5 orbitals).
    If sum_k.rot_mat is non-identity, we have to consider rotating the interaction
    Hamiltonian: the Kanamori Hamiltonian does not change because it is invariant
    under orbital mixing but all the other Hamiltonians are at most invariant
    under rotations in space.
    """
    h_int = [None] * sum_k.n_inequiv_shells

    for icrsh in range(sum_k.n_inequiv_shells):
        # ish points to the shell representative of the current group
        ish = sum_k.inequiv_to_corr[icrsh]
        n_orb = sum_k.corr_shells[ish]['dim']
        orb_names = list(range(n_orb))

        # Checks for unphysical/unimplemented combinations of orbitals and Hamiltonians
        if n_orb in (2, 3) and general_parameters['h_int_type'] != 2:
            raise NotImplementedError('Only the Kanamori Hamiltonian is '
                                      + 'implemented for the eg or t2g subset.')
        elif n_orb == 5 and general_parameters['h_int_type'] == 2:
            raise ValueError('The Kanamori Hamiltonian cannot be used for the full'
                             + 'd shell. It only applies to the eg or t2g subset.')

        # Constructs U matrix
        if n_orb in (2, 3): # e_g or t_2g cases
            Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=general_parameters['U'][icrsh],
                                            J_hund=general_parameters['J'][icrsh])
        elif n_orb == 5:
            # construct full spherical symmetric U matrix and transform to cubic basis
            # the order for the cubic orbitals is as follows ("xy","yz","z^2","xz","x^2-y^2")
            # this is consistent with the order of orbitals in the VASP interface
            # but not necessarily with wannier90!
            Umat_full = U_matrix(l=2, U_int=general_parameters['U'][icrsh],
                                 J_hund=general_parameters['J'][icrsh], basis='cubic')

            if mpi.is_master_node():
                print('\nNote: The input parameters U and J here are orbital-averaged parameters.\n'
                      + 'Same definition of U and J as in DFT+U, see\n'
                      + 'https://cms.mpi.univie.ac.at/wiki/index.php/LDAUTYPE.\n'
                      + 'WARNING: Each orbital is treated differently. Make sure that the\n'
                      + 'order of input orbitals corresponds to the order in U_matrix, see\n'
                      + 'https://triqs.github.io/triqs/2.1.x/reference/operators/util/U_matrix.html'
                      + '#pytriqs.operators.util.U_matrix.spherical_to_cubic\n')

                slater_integrals = U_J_to_radial_integrals(l=2, U_int=general_parameters['U'][icrsh],
                                                           J_hund=general_parameters['J'][icrsh])
                print('The corresponding slater integrals are')
                print('[F0, F2, F4] = [{:.2f}, {:.2f}, {:.2f}]\n'.format(*slater_integrals))
        else:
            raise NotImplementedError('Hamiltonian for n_orb = {} NOT supported'.format(n_orb))


        # Construct Hamiltonian
        mpi.report('Constructing the interaction Hamiltonian for shell {}'.format(icrsh))
        if general_parameters['h_int_type'] == 1:
            # 1. density-density
            mpi.report('Using the density-density Hamiltonian')

            # Transposes rotation matrix here because TRIQS has a slightly different definition
            Umat_full_rotated = transform_U_matrix(Umat_full, sum_k.rot_mat[ish].T)
            if not np.allclose(Umat_full_rotated, Umat_full):
                mpi.report('WARNING: applying a rotation matrix changes the dens-dens Hamiltonian.\n'
                           + 'This changes the definition of the ignored spin flip and pair hopping.')

            Umat, Upmat = reduce_4index_to_2index(Umat_full_rotated)
            h_int[icrsh] = h_int_density(general_parameters['spin_names'], orb_names,
                                         map_operator_structure=sum_k.sumk_to_solver[icrsh],
                                         U=Umat, Uprime=Upmat, H_dump=os.path.join(general_parameters['jobname'], 'H.txt'))
        elif general_parameters['h_int_type'] == 2:
            # 2. Kanamori Hamiltonian
            mpi.report('Using the Kanamori Hamiltonian (with spin-flip and pair-hopping)')
            h_int[icrsh] = h_int_kanamori(general_parameters['spin_names'], orb_names,
                                          map_operator_structure=sum_k.sumk_to_solver[icrsh],
                                          off_diag=True, U=Umat, Uprime=Upmat, J_hund=general_parameters['J'][icrsh],
                                          H_dump=os.path.join(general_parameters['jobname'], 'H.txt'))
        elif general_parameters['h_int_type'] == 3:
            # 3. Rotationally-invariant Slater Hamiltonian (4-index)
            Umat_full_rotated = transform_U_matrix(Umat_full, sum_k.rot_mat[ish].T)
            if not np.allclose(Umat_full_rotated, Umat_full):
                mpi.report('WARNING: applying a rotation matrix changes the interaction Hamiltonian.\n'
                           + 'Please be sure that the rotation is correct!')

            h_int[icrsh] = h_int_slater(general_parameters['spin_names'], orb_names,
                                        map_operator_structure=sum_k.sumk_to_solver[icrsh],
                                        off_diag=True, U_matrix=Umat_full_rotated,
                                        H_dump=os.path.join(general_parameters['jobname'], 'H.txt'))


    return h_int


def _determine_afm_mapping(general_parameters, archive, n_inequiv_shells):
    """
    Determines the symmetries that are used in AFM calculations. These
    symmetries can then be used to copy the self-energies from one impurity to
    another by exchanging up/down channels for speedup and accuracy.
    """

    afm_mapping = None
    if mpi.is_master_node():
        # Reads mapping from h5 archive if it exists already from a previous run
        if 'afm_mapping' in archive['DMFT_input']:
            afm_mapping = archive['DMFT_input']['afm_mapping']
        elif len(general_parameters['magmom']) == n_inequiv_shells:
            # find equal or opposite spin imps, where we use the magmom array to
            # identity those with equal numbers or opposite
            # [copy Yes/False, from where, switch up/down channel]
            afm_mapping = [None] * n_inequiv_shells
            abs_moms = np.abs(general_parameters['magmom'])

            for icrsh in range(n_inequiv_shells):
                # if the moment was seen before ...
                previous_occurences = np.nonzero(np.isclose(abs_moms[:icrsh], abs_moms[icrsh]))[0]
                if previous_occurences.size > 0:
                    # find the source imp to copy from
                    source = np.min(previous_occurences)
                    # determine if we need to switch up and down channel
                    switch = np.isclose(general_parameters['magmom'][icrsh], -general_parameters['magmom'][source])

                    afm_mapping[icrsh] = [True, source, switch]
                else:
                    afm_mapping[icrsh] = [False, icrsh, False]


            print('AFM calculation selected, mapping self energies as follows:')
            print('imp  [copy sigma, source imp, switch up/down]')
            print('---------------------------------------------')
            for i, elem in enumerate(afm_mapping):
                print('{}: {}'.format(i, elem))
            print('')

            archive['DMFT_input']['afm_mapping'] = afm_mapping

        # if anything did not work set afm_order false
        else:
            print('WARNING: couldn\'t determine afm mapping. No mapping used.')
            general_parameters['afm_order'] = False

    general_parameters['afm_order'] = mpi.bcast(general_parameters['afm_order'])
    if general_parameters['afm_order']:
        general_parameters['afm_mapping'] = mpi.bcast(afm_mapping)

    return general_parameters

def _determine_dc_and_initial_sigma(general_parameters, advanced_parameters, sum_k,
                                    archive, iteration_offset, density_mat_dft, solvers):
    """
    Determines the double counting (DC) and the initial Sigma. This can happen
    in five different ways.
    - Calculation resumed: use the previous DC and the Sigma of the last
        complete calculation.
    - Calculation started from previous_file: use the DC and Sigma from the
        previous file.
    - Calculation initialized with load_sigma: same as for previous_file.
        Additionally, if the DC changed (and therefore the Hartree shift), the
        initial Sigma is adjusted by that.
    - New calculation, with DC: calculate the DC, then initialize the Sigma as
        the DC, effectively starting the calculation from the DFT Green's
        function. Also breaks magnetic symmetry if calculation is magnetic.
    - New calculation, without DC: Sigma is initialized as 0, starting the
        calculation from the DFT Green's function.

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    advanced_parameters : dict
        advanced parameters as a dict
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    archive : HDFArchive
        the archive of the current calculation
    iteration_offset : int
        the iterations done before this calculation
    density_mat_dft : numpy array
        DFT density matrix
    solvers : list
        list of Solver instances

    Returns
    -------
    sum_k : SumkDFT object
        the SumkDFT object, updated by the initial Sigma and the DC
    solvers : list
        list of Solver instances, updated by the initial Sigma

    """
    start_sigma = None
    if mpi.is_master_node():
        # Resumes previous calculation
        if iteration_offset > 0:
            # TODO: python3: change to print(..., end=' ')
            print('From previous calculation:'),
            start_sigma, sum_k.dc_imp, sum_k.dc_energ, _ = toolset.load_sigma_from_h5(archive, -1)

            if general_parameters['csc'] and not general_parameters['dc_dmft']:
                sum_k = _calculate_double_counting(sum_k, density_mat_dft, general_parameters, advanced_parameters)
        # Series of calculations, loads previous sigma
        elif general_parameters['previous_file'] != 'none':
            # TODO: python3: change to print(..., end=' ')
            print('From {}:'.format(general_parameters['previous_file'])),
            with HDFArchive(general_parameters['previous_file'], 'r') as previous_archive:
                start_sigma, sum_k.dc_imp, sum_k.dc_energ, _ = toolset.load_sigma_from_h5(previous_archive, -1)
        # Loads Sigma from different calculation
        elif general_parameters['load_sigma']:
            # TODO: python3: change to print(..., end=' ')
            print('From {}:'.format(general_parameters['path_to_sigma'])),
            with HDFArchive(general_parameters['path_to_sigma'], 'r') as sigma_archive:
                (loaded_sigma, loaded_dc_imp,
                 _, loaded_density_matrix) = toolset.load_sigma_from_h5(sigma_archive,
                                                                        general_parameters['load_sigma_iter'])

            # Recalculate double counting in case U, J or DC formula changed
            if general_parameters['dc_dmft']:
                sum_k = _calculate_double_counting(sum_k, loaded_density_matrix,
                                                   general_parameters, advanced_parameters)
            else:
                sum_k = _calculate_double_counting(sum_k, density_mat_dft,
                                                   general_parameters, advanced_parameters)

            start_sigma = _set_loaded_sigma(sum_k, loaded_sigma, loaded_dc_imp)

        # Sets DC as Sigma because no initial Sigma given
        elif general_parameters['dc']:
            sum_k = _calculate_double_counting(sum_k, density_mat_dft, general_parameters, advanced_parameters)

            start_sigma = [None] * sum_k.n_inequiv_shells
            for icrsh in range(sum_k.n_inequiv_shells):
                start_sigma[icrsh] = solvers[icrsh].Sigma_iw.copy()
                dc_value = sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]]['up'][0, 0]

                if not general_parameters['csc'] and general_parameters['magnetic'] and general_parameters['magmom']:
                        # if we are doing a magnetic calculation and initial magnetic moments
                        # are set, manipulate the initial sigma accordingly
                        fac = general_parameters['magmom'][icrsh]

                        # init self energy according to factors in magmoms
                        # if magmom positive the up channel will be favored
                        for spin_channel in sum_k.gf_struct_solver[icrsh].keys():
                            if 'up' in spin_channel:
                                start_sigma[icrsh][spin_channel] << (1+fac)*dc_value
                            else:
                                start_sigma[icrsh][spin_channel] << (1-fac)*dc_value
                else:
                    start_sigma[icrsh] << dc_value
        # Sets Sigma to zero because neither initial Sigma nor DC given
        else:
            start_sigma = [solvers[icrsh].Sigma_iw.copy() for icrsh in range(sum_k.n_inequiv_shells)]
            [start_sigma_per_imp.zero() for start_sigma_per_imp in start_sigma]

    # Adds random, frequency-independent noise in zeroth iteration to break symmetries
    if not np.isclose(general_parameters['noise_level_initial_sigma'], 0) and iteration_offset == 0:
        if mpi.is_master_node():
            for start_sigma_per_imp in start_sigma:
                for _, block in start_sigma_per_imp:
                    noise = np.random.normal(scale=general_parameters['noise_level_initial_sigma'],
                                             size=block.data.shape[1:])
                    # Makes the noise hermitian
                    noise = np.broadcast_to(.5 * (noise + noise.T), block.data.shape)
                    block += GfImFreq(indices=block.indices, mesh=block.mesh, data=noise)

    # bcast everything to other nodes
    sum_k.dc_imp = mpi.bcast(sum_k.dc_imp)
    sum_k.dc_energ = mpi.bcast(sum_k.dc_energ)
    start_sigma = mpi.bcast(start_sigma)
    # Loads everything now to the solver
    for icrsh in range(sum_k.n_inequiv_shells):
        solvers[icrsh].Sigma_iw = start_sigma[icrsh]

    # Updates the sum_k object with the Matsubara self-energy
    # Symmetrizes Sigma
    for icrsh in range(sum_k.n_inequiv_shells):
        sum_k.symm_deg_gf(solvers[icrsh].Sigma_iw, orb=icrsh)

    sum_k.put_Sigma([solvers[icrsh].Sigma_iw for icrsh in range(sum_k.n_inequiv_shells)])
    
    return sum_k, solvers


def dmft_cycle(general_parameters, solver_parameters, advanced_parameters, observables):
    """
    main dmft cycle that works for one shot and CSC equally

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    solver_parameters : dict
        solver parameters as a dict
    advanced_parameters : dict
        advanced parameters as a dict
    observables : dict
        current observable array for calculation

    __Returns:__
    observables : dict
        updated observable array for calculation
    """

    # create Sumk object
    if general_parameters['csc']:
        sum_k = SumkDFT(hdf_file=general_parameters['seedname']+'.h5', use_dft_blocks=False,
                        h_field=general_parameters['h_field'])
    else:
        sum_k = SumkDFT(hdf_file=general_parameters['jobname']+'/'+general_parameters['seedname']+'.h5',
                        use_dft_blocks=False, h_field=general_parameters['h_field'])

    iteration_offset = 0

    # determine chemical potential for bare DFT sum_k object
    if mpi.is_master_node():
        archive = HDFArchive(general_parameters['jobname']+'/'+general_parameters['seedname']+'.h5', 'a')
        if not 'DMFT_results' in archive:
            archive.create_group('DMFT_results')
        if not 'last_iter' in archive['DMFT_results']:
            archive['DMFT_results'].create_group('last_iter')
        if not 'DMFT_input' in archive:
            archive.create_group('DMFT_input')
            archive['DMFT_input'].create_group('solver')
        if 'iteration_count' in archive['DMFT_results']:
            iteration_offset = archive['DMFT_results/iteration_count']
            # Backwards compatibility when chemical_potential_post was called chemical_potential
            if 'chemical_potential' in archive['DMFT_results/last_iter']:
                sum_k.chemical_potential = archive['DMFT_results/last_iter/chemical_potential']
            else:
                sum_k.chemical_potential = archive['DMFT_results/last_iter/chemical_potential_post']
    else:
        archive = None

    iteration_offset = mpi.bcast(iteration_offset)
    sum_k.chemical_potential = mpi.bcast(sum_k.chemical_potential)

    # Sets the chemical potential of the DFT calculation
    # Either directly from general parameters, if given, ...
    if general_parameters['dft_mu'] != 'none':
        dft_mu = general_parameters['dft_mu']
        # Initializes chemical potential with dft_mu if this is the first iteration
        if iteration_offset == 0:
            sum_k.chemical_potential = dft_mu
            mpi.report('\n chemical potential set to {:.3f} eV\n'.format(sum_k.chemical_potential))
    # ... or with sum_k.calc_mu
    else:
        dft_mu = sum_k.calc_mu(precision=general_parameters['prec_mu'])

    # determine block structure for solver
    det_blocks = True
    shell_multiplicity = []
    deg_shells = []
    # load previous block_structure if possible
    if mpi.is_master_node():
        if 'block_structure' in archive['DMFT_input']:
            det_blocks = False
            shell_multiplicity = archive['DMFT_input/shell_multiplicity']
            deg_shells = archive['DMFT_input/deg_shells']
    det_blocks = mpi.bcast(det_blocks)
    deg_shells = mpi.bcast(deg_shells)
    shell_multiplicity = mpi.bcast(shell_multiplicity)

    # Generates a rotation matrix to change the basis
    if (general_parameters['set_rot'] != 'none' and iteration_offset == 0
            and not general_parameters['load_sigma']):
        # calculate new rotation matrices
        sum_k = _calculate_rotation_matrix(general_parameters, sum_k, iteration_offset)
    # Saves rotation matrix to h5 archive:
    if mpi.is_master_node() and iteration_offset == 0:
        archive['DMFT_input']['rot_mat'] = sum_k.rot_mat

    mpi.barrier()

    # determine block structure for GF and Hyb function
    if det_blocks and not general_parameters['load_sigma']:
        sum_k, shell_multiplicity = toolset.determine_block_structure(sum_k, general_parameters)
    # if load sigma we need to load everything from this h5 archive
    elif general_parameters['load_sigma']:
        deg_shells = []
        # loading shell_multiplicity
        if mpi.is_master_node():
            with HDFArchive(general_parameters['path_to_sigma'], 'r') as old_calc:
                shell_multiplicity = old_calc['DMFT_input/shell_multiplicity']
                deg_shells = old_calc['DMFT_input/deg_shells']
        shell_multiplicity = mpi.bcast(shell_multiplicity)
        deg_shells = mpi.bcast(deg_shells)
        #loading block_struc and rot mat
        sum_k_old = SumkDFT(hdf_file=general_parameters['path_to_sigma'])
        sum_k_old.read_input_from_hdf(subgrp='DMFT_input', things_to_read=['block_structure', 'rot_mat'])
        sum_k.block_structure = sum_k_old.block_structure
        if not general_parameters['csc'] and general_parameters['magnetic']:
            sum_k.deg_shells = [[] for _ in range(sum_k.n_inequiv_shells)]
        else:
            sum_k.deg_shells = deg_shells
        sum_k.rot_mat = sum_k_old.rot_mat
    else:
        sum_k.read_input_from_hdf(subgrp='DMFT_input', things_to_read=['block_structure', 'rot_mat'])
        sum_k.deg_shells = deg_shells

    # Initializes empty Sigma for calculation of DFT density
    zero_Sigma_iw = [sum_k.block_structure.create_gf(ish=iineq, beta=general_parameters['beta'])
                     for iineq in range(sum_k.n_inequiv_shells)]
    sum_k.put_Sigma(zero_Sigma_iw)

    # print block structure!
    toolset.print_block_sym(sum_k)

    # extract free lattice greens function
    G_loc_all_dft = sum_k.extract_G_loc(with_Sigma=False, mu=dft_mu)
    density_mat_dft = [G_loc_all_dft[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]
    for iineq in range(sum_k.n_inequiv_shells):
        density_shell_dft = G_loc_all_dft[iineq].total_density()
        mpi.report('total density for imp {} from DFT: {:10.6f}'.format(iineq, np.real(density_shell_dft)))

    if not general_parameters['csc'] and general_parameters['magnetic']:
        sum_k.SP = 1

        if general_parameters['afm_order']:
            general_parameters = _determine_afm_mapping(general_parameters, archive, sum_k.n_inequiv_shells)

    # Initializes the solvers
    solvers = [None] * sum_k.n_inequiv_shells
    for icrsh in range(sum_k.n_inequiv_shells):
        ####################################
        # hotfix for new triqs 2.0 gf_struct_solver is still a dict
        # but cthyb 2.0 expects a list of pairs ####
        gf_struct = [[k, v] for k, v in sum_k.gf_struct_solver[icrsh].items()]
        ####################################
        # Construct the Solver instances
        if solver_parameters['measure_G_l']:
            solvers[icrsh] = Solver(beta=general_parameters['beta'], gf_struct=gf_struct,
                                    n_l=general_parameters['n_LegCoeff'])
        else:
            solvers[icrsh] = Solver(beta=general_parameters['beta'], gf_struct=gf_struct)

    # Extracts U and J
    mpi.report('*** interaction parameters ***')
    for param_name in ('U', 'J'):
        general_parameters = _extract_U_J_list(param_name, sum_k.n_inequiv_shells, general_parameters)
    for param_name in ('dc_U', 'dc_J'):
        advanced_parameters = _extract_U_J_list(param_name, sum_k.n_inequiv_shells, advanced_parameters)

    # Constructs the interaction Hamiltonian. Needs to come after setting sum_k.rot_mat
    h_int = _construct_interaction_hamiltonian(sum_k, general_parameters)
    # Saves h_int to h5 archive
    if mpi.is_master_node():
        archive['DMFT_input']['h_int'] = h_int

    # If new calculation, writes input parameters and sum_k <-> solver mapping to archive
    if iteration_offset == 0:
        if mpi.is_master_node():
            archive['DMFT_input']['general_parameters'] = general_parameters
            archive['DMFT_input']['solver_parameters'] = solver_parameters
            archive['DMFT_input']['advanced_parameters'] = advanced_parameters

            archive['DMFT_input']['block_structure'] = sum_k.block_structure
            archive['DMFT_input']['deg_shells'] = sum_k.deg_shells
            archive['DMFT_input']['shell_multiplicity'] = shell_multiplicity

    # Determines initial Sigma and DC
    sum_k, solvers = _determine_dc_and_initial_sigma(general_parameters, advanced_parameters, sum_k,
                                                     archive, iteration_offset, density_mat_dft, solvers)

    # Updates the sum_k object with the chemical potential
    if general_parameters['fixed_mu_value'] == 'none' and iteration_offset % general_parameters['mu_update_freq'] == 0:
        sum_k.calc_mu(precision=general_parameters['prec_mu'])

    # Uses fixed_mu_value as chemical potential if parameter is given
    if general_parameters['fixed_mu_value'] != 'none':
        sum_k.set_mu(general_parameters['fixed_mu_value'])
        mpi.report('+++ Keeping the chemical potential fixed at {:.3f} eV +++'.format(general_parameters['fixed_mu_value']))
    # If mu won't be updated this step, overwrite with old value
    elif iteration_offset % general_parameters['mu_update_freq'] != 0:
        if mpi.is_master_node():
            # Backwards compatibility when chemical_potential_post was called chemical_potential
            if 'chemical_potential' in archive['DMFT_results/last_iter']:
                sum_k.chemical_potential = archive['DMFT_results/observables/mu'][-1]
            else:
                sum_k.chemical_potential = archive['DMFT_results/last_iter/chemical_potential_pre']
        sum_k.chemical_potential = mpi.bcast(sum_k.chemical_potential)
        mpi.report('Chemical potential not updated this step, '
                   + 'reusing loaded one of {:.3f} eV'.format(sum_k.chemical_potential))
    # Applies mu mixing to system
    elif iteration_offset > 0:
        # Reads in the occupation and chemical_potential from the last run
        density_tot = 0
        previous_mu = None
        if mpi.is_master_node():
            dens_mat_last_iteration = archive['DMFT_results/last_iter/dens_mat_post']
            for dens_mat_per_imp, factor_per_imp in zip(dens_mat_last_iteration, shell_multiplicity):
                for density_block in dens_mat_per_imp.values():
                    density_tot += factor_per_imp * np.trace(density_block)

            # Backwards compatibility when chemical_potential_post was called chemical_potential
            if 'chemical_potential' in archive['DMFT_results/last_iter']:
                previous_mu = archive['DMFT_results/observables/mu'][-1]
            else:
                previous_mu = archive['DMFT_results/last_iter/chemical_potential_pre']
        density_tot = mpi.bcast(density_tot)
        previous_mu = mpi.bcast(previous_mu)

        sum_k.chemical_potential = _mix_chemical_potential(general_parameters, density_tot,
                                                           sum_k.density_required,
                                                           previous_mu, sum_k.chemical_potential)

    # setup of measurement of chi(SzSz(tau) if requested
    if general_parameters['measure_chi_SzSz']:
        solver_parameters, Sz_list = toolset.chi_SzSz_setup(sum_k, general_parameters, solver_parameters)
    else:
        Sz_list = None
    # if CSC we use the first and per params for the number of iterations
    if general_parameters['csc'] and iteration_offset == 0:
        # for CSC calculations converge first to a better Sigma
        n_iter = general_parameters['n_iter_dmft_first']

    # if CSC and not the first iteration we do n_iter_dmft_per iterations per DMFT step
    if general_parameters['csc'] and iteration_offset != 0:
        n_iter = general_parameters['n_iter_dmft_per']

    # if no csc calculation is done the number of dmft iterations is n_iter_dmft
    if not general_parameters['csc']:
        n_iter = general_parameters['n_iter_dmft']

    mpi.report('\n {} DMFT cycles requested. Starting with iteration  {}.\n'.format(n_iter, iteration_offset+1))
    # make sure a last time that every node as the same number of iterations
    n_iter = mpi.bcast(n_iter)

    # calculate E_kin_dft for one shot calculations
    if not general_parameters['csc'] and general_parameters['calc_energies']:
        E_kin_dft = calc_dft_kin_en(general_parameters, sum_k, dft_mu)
    else:
        E_kin_dft = None

    if mpi.is_master_node() and iteration_offset == 0:
        write_header_to_file(general_parameters, sum_k)
        observables = add_dft_values_as_zeroth_iteration(observables, general_parameters, dft_mu, sum_k,
                                                         G_loc_all_dft, density_mat_dft, shell_multiplicity)
        write_obs(observables, sum_k, general_parameters)

    converged, observables, sum_k = _dmft_steps(iteration_offset, n_iter, sum_k, solvers,
                                                general_parameters, solver_parameters, advanced_parameters,
                                                h_int, archive, shell_multiplicity, E_kin_dft,
                                                observables, Sz_list)

    # for one-shot calculations, we can use the updated GAMMA file for postprocessing
    if not general_parameters['csc'] and general_parameters['oneshot_postproc_gamma_file']:
        # Write the density correction to file after the one-shot calculation
        sum_k.calc_density_correction(filename=os.path.join(general_parameters['jobname'], 'GAMMA'), dm_type='vasp')

    if converged and general_parameters['csc']:
        # is there a smoother way to stop both vasp and triqs from running after convergency is reached?
        # TODO: this does not close vasp properly, make the same as in csc_flow
        if mpi.is_master_node():
            with open('STOPCAR', 'wt') as f_stop:
                f_stop.write('LABORT = .TRUE.\n')
            del archive
        mpi.MPI.COMM_WORLD.Abort(1)

    mpi.barrier()

    # close the h5 archive
    if mpi.is_master_node():
        if 'archive' in locals():
            del archive

    return observables

def _dmft_steps(iteration_offset, n_iter, sum_k, solvers,
                general_parameters, solver_parameters, advanced_parameters,
                h_int, archive, shell_multiplicity, E_kin_dft,
                observables, Sz_list):
    """
    Contains the actual dmft steps when all the preparation is done
    """

    sampling = False
    converged = False
    it = iteration_offset+1
    # The infamous DMFT self consistency cycle
    while it <= iteration_offset + n_iter:
        mpi.report('#'*80)
        mpi.report('Running iteration: '+str(it)+' / '+str(iteration_offset+n_iter))

        # init local density matrices for observables
        density_tot = 0.0
        density_shell = np.zeros(sum_k.n_inequiv_shells)
        density_mat = [None] * sum_k.n_inequiv_shells
        density_shell_pre = np.zeros(sum_k.n_inequiv_shells)
        density_mat_pre = [None] * sum_k.n_inequiv_shells

        mpi.barrier()
        for icrsh in range(sum_k.n_inequiv_shells):
            sum_k.symm_deg_gf(solvers[icrsh].Sigma_iw, orb=icrsh)

        # Extracts G local
        G_loc_all = sum_k.extract_G_loc()

        # Copies Sigma before Solver run for mixing later
        Sigma_iw_previous = [solvers[iineq].Sigma_iw.copy() for iineq in range(sum_k.n_inequiv_shells)]

        # looping over inequiv shells and solving for each site seperately
        for icrsh in range(sum_k.n_inequiv_shells):
            # copy the block of G_loc into the corresponding instance of the impurity solver
            solvers[icrsh].G_iw << G_loc_all[icrsh]

            density_shell_pre[icrsh] = np.real(solvers[icrsh].G_iw.total_density())
            mpi.report('\n *** Correlated Shell type #{:3d} : '.format(icrsh)
                       + 'Total charge of impurity problem = {:.6f}'.format(density_shell_pre[icrsh]))
            density_mat_pre[icrsh] = solvers[icrsh].G_iw.density()
            mpi.report('Density matrix:')
            for key, value in sorted(density_mat_pre[icrsh].items()):
                mpi.report(key)
                mpi.report(np.real(value))

            # dyson equation to extract G0_iw, using Hermitian symmetry
            solvers[icrsh].G0_iw << inverse(solvers[icrsh].Sigma_iw + inverse(solvers[icrsh].G_iw))

            solvers[icrsh].G0_iw << make_hermitian(solvers[icrsh].G0_iw)
            sum_k.symm_deg_gf(solvers[icrsh].G0_iw, orb=icrsh)

            # prepare our G_tau and G_l used to save the 'good' G_tau
            glist_tau = []
            for _, g in solvers[icrsh].G0_iw:
                glist_tau.append(GfImTime(indices=g.indices,
                                          beta=general_parameters['beta'],
                                          n_points=solvers[icrsh].n_tau))

            # we will call it G_tau_orig to store original G_tau
            solvers[icrsh].G_tau_orig = BlockGf(name_list=sum_k.gf_struct_solver[icrsh].keys(),
                                                block_list=glist_tau, make_copies=True)

            if (solver_parameters['measure_G_l']
                    or not solver_parameters['perform_tail_fit'] and general_parameters['legendre_fit']):
                glist_l = []
                for _, g in solvers[icrsh].G0_iw:
                    glist_l.append(GfLegendre(indices=g.indices,
                                              beta=general_parameters['beta'],
                                              n_points=general_parameters['n_LegCoeff']))

                solvers[icrsh].G_l_man = BlockGf(name_list=sum_k.gf_struct_solver[icrsh].keys(),
                                                 block_list=glist_l, make_copies=True)


             # store solver to h5 archive
            if general_parameters['store_solver'] and mpi.is_master_node():
                archive['DMFT_input/solver'].create_group('it_'+str(it))
                archive['DMFT_input/solver/it_'+str(it)]['S_'+str(icrsh)] = solvers[icrsh]

            # store DMFT input directly in last_iter
            if mpi.is_master_node():
                archive['DMFT_results/last_iter']['G0_iw_{}'.format(icrsh)] = solvers[icrsh].G0_iw

            # setup of measurement of chi(SzSz(tau) if requested
            if general_parameters['measure_chi_SzSz']:
                solver_parameters['measure_O_tau'] = (Sz_list[icrsh],Sz_list[icrsh])

            # if we do a AFM calculation we can use the init magnetic moments to
            # copy the self energy instead of solving it explicitly
            if (not general_parameters['csc'] and general_parameters['magnetic']
                and general_parameters['afm_order'] and general_parameters['afm_mapping'][icrsh][0]):
                imp_source = general_parameters['afm_mapping'][icrsh][1]
                invert_spin = general_parameters['afm_mapping'][icrsh][2]
                mpi.report('\ncopying the self-energy for shell {} from shell {}'.format(icrsh, imp_source))
                mpi.report('inverting spin channels: '+str(invert_spin))

                if invert_spin:
                    for spin_channel in sum_k.gf_struct_solver[icrsh].keys():
                        if 'up' in spin_channel:
                            target_channel = 'down'+spin_channel.replace('up', '')
                        else:
                            target_channel = 'up'+spin_channel.replace('down', '')

                        solvers[icrsh].Sigma_iw[spin_channel] << solvers[imp_source].Sigma_iw[target_channel]
                        solvers[icrsh].G_tau_orig[spin_channel] << solvers[imp_source].G_tau_orig[target_channel]
                        solvers[icrsh].G_iw[spin_channel] << solvers[imp_source].G_iw[target_channel]
                        solvers[icrsh].G0_iw[spin_channel] << solvers[imp_source].G0_iw[target_channel]
                        if (solver_parameters['measure_G_l']
                                or not solver_parameters['perform_tail_fit'] and general_parameters['legendre_fit']):
                            solvers[icrsh].G_l_man[spin_channel] << solvers[imp_source].G_l_man[target_channel]

                else:
                    solvers[icrsh].Sigma_iw << solvers[imp_source].Sigma_iw
                    solvers[icrsh].G_tau_orig << solvers[imp_source].G_tau_orig
                    solvers[icrsh].G_iw << solvers[imp_source].G_iw
                    solvers[icrsh].G0_iw << solvers[imp_source].G0_iw
                    if solver_parameters['measure_G_l']:
                        solvers[icrsh].G_l_man << solvers[imp_source].G_l_man

            else:
                ####################################################################
                # Solve the impurity problem for this shell
                mpi.report('\nSolving the impurity problem for shell {} ...'.format(icrsh))
                # *************************************
                solvers[icrsh].solve(h_int=h_int[icrsh], **solver_parameters)
                # *************************************
                ####################################################################

                # use Legendre for next G and Sigma instead of matsubara, less noisy!
                if solver_parameters['measure_G_l']:
                    solvers[icrsh].Sigma_iw_orig = solvers[icrsh].Sigma_iw.copy()
                    solvers[icrsh].G_iw_from_leg = solvers[icrsh].G_iw.copy()
                    solvers[icrsh].G_l_man << solvers[icrsh].G_l
                    if mpi.is_master_node():
                        for i, g in solvers[icrsh].G_l:
                            g.enforce_discontinuity(np.identity(g.target_shape[0]))
                            solvers[icrsh].G_iw[i].set_from_legendre(g)
                            # update G_tau as well:
                            solvers[icrsh].G_tau_orig[i] << Fourier(solvers[icrsh].G_iw[i])
                        # Symmetrize
                        solvers[icrsh].G_iw << make_hermitian(solvers[icrsh].G_iw)
                        # set Sigma and G_iw from G_l
                        solvers[icrsh].Sigma_iw << inverse(solvers[icrsh].G0_iw) - inverse(solvers[icrsh].G_iw)

                    # broadcast new G, Sigmas to all other nodes
                    solvers[icrsh].Sigma_iw_orig << mpi.bcast(solvers[icrsh].Sigma_iw_orig)
                    solvers[icrsh].Sigma_iw << mpi.bcast(solvers[icrsh].Sigma_iw)
                    solvers[icrsh].G_iw << mpi.bcast(solvers[icrsh].G_iw)
                    solvers[icrsh].G_tau_orig << mpi.bcast(solvers[icrsh].G_tau_orig)
                else:
                    solvers[icrsh].G_tau_orig << solvers[icrsh].G_tau
                    solvers[icrsh].G_iw << make_hermitian(solvers[icrsh].G_iw)

                    if not solver_parameters['perform_tail_fit'] and general_parameters['legendre_fit']:
                        solvers[icrsh].Sigma_iw_orig = solvers[icrsh].Sigma_iw.copy()
                        solvers[icrsh].G_iw_from_leg = solvers[icrsh].G_iw.copy()
                        solvers[icrsh].G_tau_orig = solvers[icrsh].G_tau.copy()

                        if mpi.is_master_node():
                            # run the filter
                            solvers[icrsh].G_l_man << toolset.legendre_filter(solvers[icrsh].G_tau,
                                                                              general_parameters['n_LegCoeff'])

                            # create new G_iw and G_tau
                            for i, g in solvers[icrsh].G_l_man:
                                solvers[icrsh].G_iw[i].set_from_legendre(g)
                                # update G_tau as well:
                                solvers[icrsh].G_tau_orig[i] << Fourier(solvers[icrsh].G_iw[i])
                            # Symmetrize
                            solvers[icrsh].G_iw << make_hermitian(solvers[icrsh].G_iw)
                            # set Sigma and G_iw from G_l
                            solvers[icrsh].Sigma_iw << inverse(solvers[icrsh].G0_iw) - inverse(solvers[icrsh].G_iw)


                        # broadcast new G, Sigmas to all other nodes
                        solvers[icrsh].Sigma_iw << mpi.bcast(solvers[icrsh].Sigma_iw)
                        solvers[icrsh].G_l_man << mpi.bcast(solvers[icrsh].G_l_man)
                        solvers[icrsh].G_iw << mpi.bcast(solvers[icrsh].G_iw)
                        solvers[icrsh].G_tau_orig << mpi.bcast(solvers[icrsh].G_tau_orig)


            # some printout of the obtained density matrices and some basic checks
            density_shell[icrsh] = np.real(solvers[icrsh].G_iw.total_density())
            density_tot += density_shell[icrsh]*shell_multiplicity[icrsh]
            density_mat[icrsh] = solvers[icrsh].G_iw.density()
            if mpi.is_master_node():
                print('\nTotal charge of impurity problem: {:7.5f}'.format(density_shell[icrsh]))
                print('Total charge convergency of impurity problem: {:7.5f}'.format(density_shell[icrsh]-density_shell_pre[icrsh]))
                print('\nDensity matrix:')
                for key, value in sorted(density_mat[icrsh].items()):
                    value = np.real(value)
                    print(key)
                    print(value)
                    eigenvalues = np.linalg.eigvalsh(value)
                    print('eigenvalues: {}'.format(eigenvalues))
                    # check for large off-diagonal elements and write out a warning
                    if np.max(np.abs(value - np.diag(np.diag(value)))) >= 0.1:
                        print('\n!!! WARNING !!!')
                        print('!!! large off diagonal elements in density matrix detected! I hope you know what you are doing !!!')
                        print('!!! WARNING !!!\n')

        # Done with loop over impurities

        if mpi.is_master_node():
            # Done. Now do post-processing:
            print('\n *** Post-processing the solver output ***')
            print('Total charge of all correlated shells : {:.6f}\n'.format(density_tot))

        # mixing Sigma
        if mpi.is_master_node():
            print('mixing sigma with previous iteration by factor {:.3f}\n'.format(general_parameters['sigma_mix']))
            for icrsh in range(sum_k.n_inequiv_shells):
                solvers[icrsh].Sigma_iw << (general_parameters['sigma_mix'] * solvers[icrsh].Sigma_iw
                                            + (1-general_parameters['sigma_mix']) * Sigma_iw_previous[icrsh])

        for icrsh in range(sum_k.n_inequiv_shells):
            solvers[icrsh].Sigma_iw << mpi.bcast(solvers[icrsh].Sigma_iw)
        mpi.barrier()

        # calculate new DC
        if general_parameters['dc'] and general_parameters['dc_dmft']:
            sum_k = _calculate_double_counting(sum_k, density_mat,
                                               general_parameters, advanced_parameters)

        # symmetrise Sigma
        for icrsh in range(sum_k.n_inequiv_shells):
            sum_k.symm_deg_gf(solvers[icrsh].Sigma_iw, orb=icrsh)

        # doing the dmft loop and set new sigma into sumk
        sum_k.put_Sigma([solvers[icrsh].Sigma_iw for icrsh in range(sum_k.n_inequiv_shells)])

        # saving previous mu for writing to observables file
        previous_mu = sum_k.chemical_potential

        if general_parameters['fixed_mu_value'] != 'none':
            sum_k.set_mu(general_parameters['fixed_mu_value'])
            mpi.report('+++ Keeping the chemical potential fixed at {:.3f} eV +++'.format(general_parameters['fixed_mu_value']))
        else:
            # Updates chemical potential every mu_update_freq iterations
            if it % general_parameters['mu_update_freq'] == 0:
                sum_k.calc_mu(precision=general_parameters['prec_mu'])
                sum_k.chemical_potential = _mix_chemical_potential(general_parameters, density_tot,
                                                                   sum_k.density_required,
                                                                   previous_mu, sum_k.chemical_potential)
            else:
                mpi.report('Chemical potential not updated this step, '
                           + 'reusing previous one of {:.3f} eV'.format(sum_k.chemical_potential))

        # Saves results to h5 archive
        if mpi.is_master_node():
            # Writes all results to a dictionary that will be saved
            write_to_h5 = {'chemical_potential_post': sum_k.chemical_potential,
                           'chemical_potential_pre': previous_mu,
                           'DC_pot': sum_k.dc_imp,
                           'DC_energ': sum_k.dc_energ,
                           'dens_mat_pre': density_mat_pre,
                           'dens_mat_post': density_mat,
                          }

            for icrsh in range(sum_k.n_inequiv_shells):
                # Checks first if G_tau was set and if not the info is stored in _orig
                if solvers[icrsh].G_tau:
                    write_to_h5['Gimp_tau_{}'.format(icrsh)] = solvers[icrsh].G_tau
                    # if legendre was set, that we have both now!
                    if (solver_parameters['measure_G_l']
                        or not solver_parameters['perform_tail_fit'] and general_parameters['legendre_fit']):
                        write_to_h5['Gimp_tau_orig{}'.format(icrsh)] = solvers[icrsh].G_tau_orig
                else:
                    write_to_h5['Gimp_tau_{}'.format(icrsh)] = solvers[icrsh].G_tau_orig

                write_to_h5['G0_iw_{}'.format(icrsh)] = solvers[icrsh].G0_iw
                write_to_h5['Delta_tau_{}'.format(icrsh)] = solvers[icrsh].Delta_tau
                write_to_h5['Gimp_iw_{}'.format(icrsh)] = solvers[icrsh].G_iw
                write_to_h5['Sigma_iw_{}'.format(icrsh)] = solvers[icrsh].Sigma_iw

                if (solver_parameters['measure_G_l']
                    or not solver_parameters['perform_tail_fit'] and general_parameters['legendre_fit']):
                    write_to_h5['Gimp_l_{}'.format(icrsh)] = solvers[icrsh].G_l_man

                if solver_parameters['measure_pert_order']:
                    write_to_h5['pert_order_imp_{}'.format(icrsh)] = solvers[icrsh].perturbation_order
                    write_to_h5['pert_order_total_imp_{}'.format(icrsh)] = solvers[icrsh].perturbation_order_total

                if general_parameters['measure_chi_SzSz']:
                    write_to_h5['O_tau_{}'.format(icrsh)] = solvers[icrsh].O_tau

            # Backward compatibility: removes renamed keys if still in last_iter
            keys_to_remove = ['chemical_potential']
            keys_to_remove += ['G0_iw{}'.format(icrsh) for icrsh in range(sum_k.n_inequiv_shells)]
            keys_to_remove += ['Delta_tau{}'.format(icrsh) for icrsh in range(sum_k.n_inequiv_shells)]
            for key in keys_to_remove:
                if key in archive['DMFT_results/last_iter']:
                    del archive['DMFT_results/last_iter'][key]

            # Saves the results to last_iter
            archive['DMFT_results']['iteration_count'] = it
            for key, value in write_to_h5.items():
                archive['DMFT_results/last_iter'][key] = value

            # Permanently saves to h5 archive every h5_save_freq iterations
            if ((not sampling and it % general_parameters['h5_save_freq'] == 0)
                    or (sampling and it % general_parameters['sampling_h5_save_freq'] == 0)):
                archive['DMFT_results'].create_group('it_{}'.format(it))
                for key, value in write_to_h5.items():
                    archive['DMFT_results/it_{}'.format(it)][key] = value

        mpi.barrier()

        # if we do a CSC calculation we need always an updated GAMMA file
        E_bandcorr = 0.0
        if general_parameters['csc']:
            # handling the density correction for fcsc calculations
            E_bandcorr = sum_k.calc_density_correction(filename='GAMMA', dm_type='vasp')[2]

        # for a one shot calculation we are using our own method
        if not general_parameters['csc'] and general_parameters['calc_energies']:
            E_bandcorr = calc_bandcorr_man(general_parameters, sum_k, E_kin_dft)

        # calculate observables and write them to file
        if mpi.is_master_node():
            print('\n *** calculation of observables ***')
            observables = add_dmft_observables(observables,
                                               general_parameters,
                                               solver_parameters,
                                               it,
                                               solvers,
                                               h_int,
                                               previous_mu,
                                               sum_k,
                                               density_mat,
                                               shell_multiplicity,
                                               E_bandcorr)

            write_obs(observables, sum_k, general_parameters)

            # write the new observable array to h5 archive
            archive['DMFT_results']['observables'] = observables

            print('*** iteration finished ***')

            # print out of the energies
            if general_parameters['calc_energies']:
                print('\n' + '='*60)
                print('summary of energetics:')
                print('total energy: ', observables['E_tot'][-1])
                print('DFT energy: ', observables['E_dft'][-1])
                print('correllation energy: ', observables['E_corr_en'][-1])
                print('DFT band correction: ', observables['E_bandcorr'][-1])
                print('='*60 + '\n')

            # print out summary of occupations per impurity
            print('='*60)
            print('summary of occupations:')
            for icrsh in range(sum_k.n_inequiv_shells):
                print('total occupany of impurity '+str(icrsh)+': {:7.4f}'.format(observables['imp_occ'][icrsh]['up'][-1]+observables['imp_occ'][icrsh]['down'][-1]))
            for icrsh in range(sum_k.n_inequiv_shells):
                print('G(beta/2) occ of impurity '+str(icrsh)+': {:8.4f}'.format(observables['imp_gb2'][icrsh]['up'][-1]+observables['imp_gb2'][icrsh]['down'][-1]))
            print('='*60 + '\n')

            # if a magnetic calculation is done print out a summary of up/down occ
            if not general_parameters['csc'] and general_parameters['magnetic']:
                occ = {}
                occ['up'] = 0.0
                occ['down'] = 0.0
                print('\n' + '='*60)
                print('\n *** summary of magnetic occupations: ***')
                for icrsh in range(sum_k.n_inequiv_shells):
                    for spin in ['up', 'down']:
                        temp = observables['imp_occ'][icrsh][spin][-1]
                        print('imp '+str(icrsh)+' spin '+spin+': {:6.4f}'.format(temp))
                        occ[spin] += temp

                print('total spin up   occ: '+'{:6.4f}'.format(occ['up']))
                print('total spin down occ: '+'{:6.4f}'.format(occ['down']))
                print('='*60 + '\n')

        # check for convergency and stop if criteria is reached
        std_dev = 0.0

        if general_parameters['occ_conv_crit'] > 0.0:
            conv_file = open(general_parameters['jobname']+'/'+'convergence.dat', 'a')

        if it == 1 and general_parameters['occ_conv_crit'] > 0.0 and mpi.is_master_node():
            conv_file.write('std_dev occ for each impurity \n')
        if general_parameters['occ_conv_crit'] > 0.0 and it >= general_parameters['occ_conv_it']:
            if mpi.is_master_node():
                # if convergency criteria was already reached dont overwrite it!
                if converged:
                    _, std_dev = toolset.check_convergence(sum_k, general_parameters, observables)
                else:
                    converged, std_dev = toolset.check_convergence(sum_k, general_parameters, observables)
                conv_file.write('{:3d}'.format(it))
                for icrsh in range(sum_k.n_inequiv_shells):
                    conv_file.write('{:10.6f}'.format(std_dev[icrsh]))
                conv_file.write('\n')
                conv_file.flush()
            converged = mpi.bcast(converged)
            std_dev = mpi.bcast(std_dev)

        if mpi.is_master_node():
            if general_parameters['occ_conv_crit'] > 0.0:
                conv_file.close()

        # check for convergency and if wanted do the sampling dmft iterations.
        if converged and not sampling:
            if general_parameters['sampling_iterations'] > 0:
                mpi.report('*** required convergence reached and sampling now for '+str(general_parameters['sampling_iterations'])+' iterations ***')
                n_iter = (it-iteration_offset)+general_parameters['sampling_iterations']
                sampling = True
            else:
                mpi.report('*** required convergence reached stopping now ***')
                break

        # finishing the dmft loop if the maximum number of iterations is reached
        if it == (iteration_offset + n_iter):
            mpi.report('all requested iterations finished')
            mpi.report('#'*80)
        # check also if the global number of iterations is reached, important for CSC
        elif general_parameters['csc'] and it == general_parameters['prev_iterations'] + general_parameters['n_iter_dmft']:
            # TODO: bug, if sampling needs more iterations than n_iter_dmft, this will stop sampling too early
            mpi.report('all requested iterations finished')
            mpi.report('#'*80)
            break

        # iteration counter
        it += 1

    return converged, observables, sum_k
