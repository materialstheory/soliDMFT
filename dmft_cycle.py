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

# triqs
import pytriqs.utility.mpi as mpi
from pytriqs.operators.util.U_matrix import (U_matrix, U_matrix_kanamori, reduce_4index_to_2index,
                                             U_J_to_radial_integrals, transform_U_matrix)
from pytriqs.operators.util.hamiltonians import h_int_kanamori, h_int_density, h_int_slater
from pytriqs.archive import HDFArchive
from triqs_cthyb.solver import Solver
from triqs_dft_tools.sumk_dft import SumkDFT
from pytriqs.gf import GfImTime, GfLegendre, BlockGf, make_hermitian
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

    # Sets the DC and exits the function if advanced_parameters['dc_fixed_value'] is specified
    if advanced_parameters['dc_fixed_value'] != 'none':
        for icrsh in range(sum_k.n_inequiv_shells):
            sum_k.calc_dc(density_matrix[icrsh], orb=icrsh,
                          use_dc_value=advanced_parameters['dc_fixed_value'])
        return sum_k

    # The regular way: calculates the DC based on U, J and the dc_type
    for icrsh in range(sum_k.n_inequiv_shells):
        if general_parameters['dc_type'] == 3:
            # this is FLL for eg orbitals only as done in Seth PRB 96 205139 2017 eq 10
            # this setting for U and J is reasonable as it is in the spirit of F0 and Javg
            # for the 5 orb case
            mpi.report('Doing FLL DC for eg orbitals only with Uavg=U-J and Javg=2*J')
            Uavg = general_parameters['U'][icrsh] - general_parameters['J'][icrsh]
            Javg = 2*general_parameters['J'][icrsh]
            sum_k.calc_dc(density_matrix[icrsh], U_interact=Uavg, J_hund=Javg,
                          orb=icrsh, use_dc_formula=0)
        else:
            sum_k.calc_dc(density_matrix[icrsh], U_interact=general_parameters['U'][icrsh],
                          J_hund=general_parameters['J'][icrsh], orb=icrsh,
                          use_dc_formula=general_parameters['dc_type'])

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


def _calculate_rotation_matrix(general_parameters, sum_k, iteration_offset, density_mat_dft):
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
        q_diag = density_mat_dft
        chnl = 'up_0'
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
            # TODO: this might not be consistent with the way the W90 converter uses the rot_mat
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
    dft_mu = 0.0

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
            iteration_offset = archive['DMFT_results']['iteration_count']
            sum_k.chemical_potential = archive['DMFT_results']['last_iter']['chemical_potential']
        if general_parameters['dft_mu'] != 0.0:
            dft_mu = general_parameters['dft_mu']
    else:
        archive = None

    # cast everything to other nodes
    sum_k.chemical_potential = mpi.bcast(sum_k.chemical_potential)
    dft_mu = mpi.bcast(dft_mu)

    iteration_offset = mpi.bcast(iteration_offset)

    if iteration_offset == 0 and dft_mu != 0.0:
        sum_k.chemical_potential = dft_mu
        mpi.report('\n chemical potential set to '+str(sum_k.chemical_potential)+' eV \n')


    # determine block structure for solver
    det_blocks = True
    shell_multiplicity = []
    deg_shells = []
    # load previous block_structure if possible
    if mpi.is_master_node():
        if 'block_structure' in archive['DMFT_input']:
            det_blocks = False
            shell_multiplicity = archive['DMFT_input']['shell_multiplicity']
            deg_shells = archive['DMFT_input']['deg_shells']
    det_blocks = mpi.bcast(det_blocks)
    deg_shells = mpi.bcast(deg_shells)
    shell_multiplicity = mpi.bcast(shell_multiplicity)

    # determine true dft_mu
    dft_mu = sum_k.calc_mu(precision=general_parameters['prec_mu'])


    # determine block structure for GF and Hyb function
    if det_blocks and not general_parameters['load_sigma']:
        sum_k, shell_multiplicity = toolset.determine_block_structure(sum_k, general_parameters)
    # if load sigma we need to load everything from this h5 archive
    elif general_parameters['load_sigma']:
        deg_shells = []
        # loading shell_multiplicity
        if mpi.is_master_node():
            with HDFArchive(general_parameters['path_to_sigma'], 'r') as old_calc:
                shell_multiplicity = old_calc['DMFT_input']['shell_multiplicity']
                deg_shells = old_calc['DMFT_input']['deg_shells']
        shell_multiplicity = mpi.bcast(shell_multiplicity)
        deg_shells = mpi.bcast(deg_shells)
        #loading block_struc and rot mat
        sum_k_old = SumkDFT(hdf_file=general_parameters['path_to_sigma'])
        sum_k_old.read_input_from_hdf(subgrp='DMFT_input', things_to_read=['block_structure', 'rot_mat'])
        sum_k.block_structure = sum_k_old.block_structure
        if general_parameters['magnetic']:
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
    toolset.print_block_sym(sum_k)

    # extract free lattice greens function
    G_loc_all_dft = sum_k.extract_G_loc(with_Sigma=False, mu=dft_mu)
    density_mat_dft = [G_loc_all_dft[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]
    for iineq in range(sum_k.n_inequiv_shells):
        density_shell_dft = G_loc_all_dft[iineq].total_density()
        mpi.report('total density for imp {} from DFT: {:10.6f}'.format(iineq, np.real(density_shell_dft)))

    # for CSC calculations this does not work
    if general_parameters['magnetic'] and not general_parameters['csc']:
        sum_k.SP = 1
    # if we do AFM calculation we can use symmetry to copy self-energies from
    # one imp to another by exchanging up/down channels for speed up and accuracy
    general_parameters['afm_mapping'] = []
    if mpi.is_master_node():
        if (general_parameters['magnetic']
                and len(general_parameters['magmom']) == sum_k.n_inequiv_shells
                and general_parameters['afm_order']
                and not 'afm_mapping' in archive['DMFT_input']):

            # find equal or opposite spin imps, where we use the magmom array to
            # identity those with equal numbers or opposite
            # [copy Yes/False, from where, switch up/down channel]
            general_parameters['afm_mapping'].append([False, 0, False])

            abs_moms = map(abs, general_parameters['magmom'])

            for icrsh in range(1, sum_k.n_inequiv_shells):
                # if the moment was seen before ...
                if abs_moms[icrsh] in abs_moms[:icrsh]:
                    copy = True
                    # find the source imp to copy from
                    source = abs_moms[:icrsh].index(abs_moms[icrsh])

                    # determine if we need to switch up and down channel
                    if general_parameters['magmom'][icrsh] == general_parameters['magmom'][source]:
                        switch = False
                    elif general_parameters['magmom'][icrsh] == -1*general_parameters['magmom'][source]:
                        switch = True
                    # double check if the moment where not the same and then don't copy
                    else:
                        switch = False
                        copy = False

                    general_parameters['afm_mapping'].append([copy, source, switch])
                else:
                    general_parameters['afm_mapping'].append([False, icrsh, False])


            print('AFM calculation selected, mapping self energies as follows:')
            print('imp  [copy sigma, source imp, switch up/down]')
            print('---------------------------------------------')
            for i, elem in enumerate(general_parameters['afm_mapping']):
                print(str(i)+' ', elem)
            print('')

            archive['DMFT_input']['afm_mapping'] = general_parameters['afm_mapping']

        # else if mapping is already in h5 archive
        elif 'afm_mapping' in archive['DMFT_input']:
            general_parameters['afm_mapping'] = archive['DMFT_input']['afm_mapping']

        # if anything did not work set afm_order false
        else:
            general_parameters['afm_order'] = False

    general_parameters['afm_order'] = mpi.bcast(general_parameters['afm_order'])
    general_parameters['afm_mapping'] = mpi.bcast(general_parameters['afm_mapping'])

    # Initializes the solvers
    solvers = []
    for icrsh in range(sum_k.n_inequiv_shells):
        ####################################
        # hotfix for new triqs 2.0 gf_struct_solver is still a dict
        # but cthyb 2.0 expects a list of pairs ####
        gf_struct = [[k, v] for k, v in sum_k.gf_struct_solver[icrsh].iteritems()]
        ####################################
        # Construct the Solver instances
        if solver_parameters['measure_G_l']:
            solvers.append(Solver(beta=general_parameters['beta'], gf_struct=gf_struct,
                                  n_l=general_parameters['n_LegCoeff']))
        else:
            solvers.append(Solver(beta=general_parameters['beta'], gf_struct=gf_struct))

    # if Sigma is loaded, mu needs to be calculated again
    calc_mu = False

    # extract U and J
    mpi.report('*** interaction parameters ***')
    for param_name in ('U', 'J'):
        general_parameters = _extract_U_J_list(param_name, sum_k.n_inequiv_shells, general_parameters)

    # Prepare hdf file and and check for previous iterations
    if mpi.is_master_node():
        if 'iteration_count' in archive['DMFT_results']:
            print('\n *** loading previous self energies ***')
            sum_k.dc_imp = archive['DMFT_results']['last_iter']['DC_pot']
            sum_k.dc_energ = archive['DMFT_results']['last_iter']['DC_energ']
            for icrsh in range(sum_k.n_inequiv_shells):
                print('loading Sigma_imp'+str(icrsh)+' from previous calculation')
                solvers[icrsh].Sigma_iw = archive['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)]
            calc_mu = True
        else:
            # calculation from scratch:
            ## write some input parameters to the archive
            archive['DMFT_input']['general_parameters'] = general_parameters
            archive['DMFT_input']['solver_parameters'] = solver_parameters
            archive['DMFT_input']['advanced_parameters'] = advanced_parameters

            ## and also the SumK <--> Solver mapping (used for restarting)
            for item in ['block_structure', 'deg_shells']:
                archive['DMFT_input'][item] = getattr(sum_k, item)
            # and the shell_multiplicity
            archive['DMFT_input']['shell_multiplicity'] = shell_multiplicity

            start_sigma = None
            # load now sigma from other calculation if wanted
            if general_parameters['load_sigma'] and general_parameters['previous_file'] == 'none':
                (loaded_sigma, loaded_dc_imp,
                 _, loaded_density_matrix) = toolset.load_sigma_from_h5(general_parameters['path_to_sigma'],
                                                                        general_parameters['load_sigma_iter'])

                # Recalculate double counting in case U, J or DC formula changed
                if general_parameters['dc_dmft']:
                    sum_k = _calculate_double_counting(sum_k, loaded_density_matrix,
                                                       general_parameters, advanced_parameters)
                else:
                    sum_k = _calculate_double_counting(sum_k, density_mat_dft,
                                                       general_parameters, advanced_parameters)

                start_sigma = _set_loaded_sigma(sum_k, loaded_sigma, loaded_dc_imp)

            # if this is a series of calculation load previous sigma
            elif general_parameters['previous_file'] != 'none':
                start_sigma, sum_k.dc_imp, sum_k.dc_energ, _ = toolset.load_sigma_from_h5(general_parameters['previous_file'], -1)

            # load everything now to the solver
            if start_sigma:
                calc_mu = True
                for icrsh in range(sum_k.n_inequiv_shells):
                    solvers[icrsh].Sigma_iw = start_sigma[icrsh]

    # bcast everything to other nodes
    # especially the sum_k changed things, since they have been only read
    # by the master node
    for icrsh in range(sum_k.n_inequiv_shells):
        solvers[icrsh].Sigma_iw = mpi.bcast(solvers[icrsh].Sigma_iw)
        solvers[icrsh].G_iw = mpi.bcast(solvers[icrsh].G_iw)
    sum_k.dc_imp = mpi.bcast(sum_k.dc_imp)
    sum_k.dc_energ = mpi.bcast(sum_k.dc_energ)
    sum_k.set_dc(sum_k.dc_imp, sum_k.dc_energ)
    calc_mu = mpi.bcast(calc_mu)

    # symmetrise Sigma
    for icrsh in range(sum_k.n_inequiv_shells):
        sum_k.symm_deg_gf(solvers[icrsh].Sigma_iw, orb=icrsh)

    sum_k.put_Sigma([solvers[icrsh].Sigma_iw for icrsh in range(sum_k.n_inequiv_shells)])
    if calc_mu:
        # determine chemical potential
        sum_k.calc_mu(precision=general_parameters['prec_mu'])

    if mpi.is_master_node():
        # print other system information
        print('\nInverse temperature beta = {}'.format(general_parameters['beta']))
        if solver_parameters['measure_G_l']:
            print('\nSampling G(iw) in Legendre space with {} coefficients'.format(general_parameters['n_LegCoeff']))

    # Generates a rotation matrix to change the basis
    if (general_parameters['set_rot'] != 'none' and iteration_offset == 0
            and not general_parameters['load_sigma']):
        sum_k = _calculate_rotation_matrix(general_parameters, sum_k, iteration_offset, density_mat_dft)
    # Saves rotation matrix to h5 archive:
    if mpi.is_master_node() and iteration_offset == 0:
        archive['DMFT_input']['rot_mat'] = sum_k.rot_mat

    # Constructs the interaction Hamiltonian. Needs to come after setting sum_k.rot_mat
    h_int = _construct_interaction_hamiltonian(sum_k, general_parameters)
    # Saves h_int to h5 archive
    if mpi.is_master_node():
        archive['DMFT_input']['h_int'] = h_int

    #Double Counting if first iteration or if CSC calculation with DFTDC
    if general_parameters['dc'] and ((iteration_offset == 0 and not general_parameters['load_sigma']) or
                                     (general_parameters['csc'] and not general_parameters['dc_dmft'])):
        sum_k = _calculate_double_counting(sum_k, density_mat_dft, general_parameters, advanced_parameters)

    # initialise sigma if first iteration
    if (iteration_offset == 0 and general_parameters['previous_file'] == 'none'
            and not general_parameters['load_sigma'] and general_parameters['dc']):
        for icrsh in range(sum_k.n_inequiv_shells):
            # if we are doing a mangetic calculation and initial magnetic moments
            # are set, manipulate the initial sigma accordingly
            if general_parameters['magnetic'] and general_parameters['magmom']:
                fac = abs(general_parameters['magmom'][icrsh])

                # init self energy according to factors in magmoms
                if general_parameters['magmom'][icrsh] > 0.0:
                    # if larger 1 the up channel will be favored
                    for spin_channel, elem in sum_k.gf_struct_solver[icrsh].iteritems():
                        if 'up' in spin_channel:
                            solvers[icrsh].Sigma_iw[spin_channel] << (1+fac)*sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]]['up'][0, 0]
                        else:
                            solvers[icrsh].Sigma_iw[spin_channel] << (1-fac)*sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]]['down'][0, 0]
                else:
                    for spin_channel, elem in sum_k.gf_struct_solver[icrsh].iteritems():
                        if 'down' in spin_channel:
                            solvers[icrsh].Sigma_iw[spin_channel] << (1+fac)*sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]]['up'][0, 0]
                        else:
                            solvers[icrsh].Sigma_iw[spin_channel] << (1-fac)*sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]]['down'][0, 0]
            else:
                solvers[icrsh].Sigma_iw << sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]]['up'][0, 0]

        # set DC as Sigma and extract the new Gloc with DC potential
        sum_k.put_Sigma([solvers[icrsh].Sigma_iw for icrsh in range(sum_k.n_inequiv_shells)])

    if ((general_parameters['load_sigma'] or general_parameters['previous_file'] != 'none')
            and iteration_offset == 0):
        sum_k.calc_mu(precision=general_parameters['prec_mu'])

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
            for key, value in density_mat_pre[icrsh].iteritems():
                mpi.report(key)
                mpi.report(np.real(value))

            # dyson equation to extract G0_iw, using Hermitian symmetry
            solvers[icrsh].G0_iw << inverse(solvers[icrsh].Sigma_iw + inverse(solvers[icrsh].G_iw))

            solvers[icrsh].G0_iw << make_hermitian(solvers[icrsh].G0_iw)
            sum_k.symm_deg_gf(solvers[icrsh].G0_iw, orb=icrsh)

            # prepare our G_tau and G_l used to save the 'good' G_tau
            glist_tau = []
            if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                glist_l = []
            for name, g in solvers[icrsh].G0_iw:
                glist_tau.append(GfImTime(indices=g.indices,
                                          beta=general_parameters['beta'],
                                          n_points=solvers[icrsh].n_tau))
                if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                    glist_l.append(GfLegendre(indices=g.indices,
                                              beta=general_parameters['beta'],
                                              n_points=general_parameters['n_LegCoeff']))

            # we will call it G_tau_orig to store original G_tau
            solvers[icrsh].G_tau_orig = BlockGf(name_list=sum_k.gf_struct_solver[icrsh].keys(),
                                               block_list=glist_tau, make_copies=True)

            if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                solvers[icrsh].G_l_man = BlockGf(name_list=sum_k.gf_struct_solver[icrsh].keys(),
                                                 block_list=glist_l, make_copies=True)

             # store solver to h5 archive
            if general_parameters['store_solver'] and mpi.is_master_node():
                archive['DMFT_input']['solver'].create_group('it_'+str(it))
                archive['DMFT_input']['solver']['it_'+str(it)]['S_'+str(icrsh)] = solvers[icrsh]

            # store DMFT input directly in last_iter
            if mpi.is_master_node():
                archive['DMFT_results']['last_iter']['G0_iw'+str(icrsh)] = solvers[icrsh].G0_iw

            # setup of measurement of chi(SzSz(tau) if requested
            if general_parameters['measure_chi_SzSz']:
                solver_parameters['measure_O_tau'] = (Sz_list[icrsh],Sz_list[icrsh])

            # if we do a AFM calculation we can use the init magnetic moments to
            # copy the self energy instead of solving it explicitly
            if general_parameters['afm_order'] and general_parameters['afm_mapping'][icrsh][0]:
                imp_source = general_parameters['afm_mapping'][icrsh][1]
                invert_spin = general_parameters['afm_mapping'][icrsh][2]
                mpi.report('\ncopying the self-energy for shell {} from shell {}'.format(icrsh, imp_source))
                mpi.report('inverting spin channels: '+str(invert_spin))

                if invert_spin:
                    for spin_channel, _ in sum_k.gf_struct_solver[icrsh].iteritems():
                        if 'up' in spin_channel:
                            target_channel = 'down'+spin_channel.replace('up', '')
                        else:
                            target_channel = 'up'+spin_channel.replace('down', '')

                        solvers[icrsh].Sigma_iw[spin_channel] << solvers[imp_source].Sigma_iw[target_channel]
                        solvers[icrsh].G_tau_orig[spin_channel] << solvers[imp_source].G_tau_orig[target_channel]
                        solvers[icrsh].G_iw[spin_channel] << solvers[imp_source].G_iw[target_channel]
                        solvers[icrsh].G0_iw[spin_channel] << solvers[imp_source].G0_iw[target_channel]
                        if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
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

                    if general_parameters['legendre_fit']:
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
                for key, value in density_mat[icrsh].iteritems():
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

        if general_parameters['fixed_mu_value'] != 'none':
            sum_k.set_mu(general_parameters['fixed_mu_value'])
            # TODO: put previous_mu above set_mu to make it consistent with case of no fixed mu
            previous_mu = sum_k.chemical_potential
            mpi.report('+++ Keeping the chemical potential fixed at {} eV +++'.format(general_parameters['fixed_mu_value']))
        else:
            # saving previous mu for writing to observables file
            previous_mu = sum_k.chemical_potential
            sum_k.calc_mu(precision=general_parameters['prec_mu'])

        # saving results to h5 archive
        if mpi.is_master_node():
            archive['DMFT_results']['iteration_count'] = it
            archive['DMFT_results']['last_iter']['chemical_potential'] = sum_k.chemical_potential
            archive['DMFT_results']['last_iter']['DC_pot'] = sum_k.dc_imp
            archive['DMFT_results']['last_iter']['DC_energ'] = sum_k.dc_energ
            archive['DMFT_results']['last_iter']['dens_mat_pre'] = density_mat_pre
            archive['DMFT_results']['last_iter']['dens_mat_post'] = density_mat
            for icrsh in range(sum_k.n_inequiv_shells):

                # check first if G_tau was set and if not the info is stored in _orig
                if solvers[icrsh].G_tau:
                    archive['DMFT_results']['last_iter']['Gimp_tau_'+str(icrsh)] = solvers[icrsh].G_tau
                    # if legendre was set, that we have both now!
                    if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                        archive['DMFT_results']['last_iter']['Gimp_tau_orig'+str(icrsh)] = solvers[icrsh].G_tau_orig
                else:
                    archive['DMFT_results']['last_iter']['Gimp_tau_'+str(icrsh)] = solvers[icrsh].G_tau_orig

                archive['DMFT_results']['last_iter']['Delta_tau'+str(icrsh)] = solvers[icrsh].Delta_tau
                if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                    archive['DMFT_results']['last_iter']['Gimp_l_'+str(icrsh)] = solvers[icrsh].G_l_man

                if solver_parameters['measure_pert_order']:
                    archive['DMFT_results']['last_iter']['pert_order_imp_'+str(icrsh)] = solvers[icrsh].perturbation_order
                    archive['DMFT_results']['last_iter']['pert_order_total_imp_'+str(icrsh)] = solvers[icrsh].perturbation_order_total
                archive['DMFT_results']['last_iter']['Gimp_iw_'+str(icrsh)] = solvers[icrsh].G_iw
                archive['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)] = solvers[icrsh].Sigma_iw

                if general_parameters['measure_chi_SzSz']:
                    archive['DMFT_results']['last_iter']['O_tau_'+str(icrsh)] = solvers[icrsh].O_tau

            # save to h5 archive every h5_save_freq iterations
            if it % general_parameters['h5_save_freq'] == 0:
                archive['DMFT_results'].create_group('it_'+str(it))
                archive['DMFT_results']['it_'+str(it)]['chemical_potential'] = sum_k.chemical_potential
                archive['DMFT_results']['it_'+str(it)]['DC_pot'] = sum_k.dc_imp
                archive['DMFT_results']['it_'+str(it)]['DC_energ'] = sum_k.dc_energ
                archive['DMFT_results']['it_'+str(it)]['dens_mat_pre'] = density_mat_pre
                archive['DMFT_results']['it_'+str(it)]['dens_mat_post'] = density_mat
                for icrsh in range(sum_k.n_inequiv_shells):
                    archive['DMFT_results']['it_'+str(it)]['G0_iw'+str(icrsh)] = solvers[icrsh].G0_iw
                    archive['DMFT_results']['last_iter']['Delta_tau'+str(icrsh)] = solvers[icrsh].Delta_tau
                    # check if G_tau was set
                    if solvers[icrsh].G_tau:
                        archive['DMFT_results']['it_'+str(it)]['Gimp_tau_'+str(icrsh)] = solvers[icrsh].G_tau
                        # if legendre was set, that we have both now!
                        if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                            archive['DMFT_results']['it_'+str(it)]['Gimp_tau_orig'+str(icrsh)] = solvers[icrsh].G_tau_orig
                    else:
                        archive['DMFT_results']['it_'+str(it)]['Gimp_tau_'+str(icrsh)] = solvers[icrsh].G_tau_orig

                    if solver_parameters['measure_G_l'] or general_parameters['legendre_fit']:
                        archive['DMFT_results']['it_'+str(it)]['Gimp_l_'+str(icrsh)] = solvers[icrsh].G_l_man
                    archive['DMFT_results']['it_'+str(it)]['Gimp_iw_'+str(icrsh)] = solvers[icrsh].G_iw
                    archive['DMFT_results']['it_'+str(it)]['Sigma_iw_'+str(icrsh)] = solvers[icrsh].Sigma_iw

                    if solver_parameters['measure_pert_order']:
                        archive['DMFT_results']['it_'+str(it)]['pert_order_imp_'+str(icrsh)] = solvers[icrsh].perturbation_order
                        archive['DMFT_results']['it_'+str(it)]['pert_order_total_imp_'+str(icrsh)] = solvers[icrsh].perturbation_order_total

                    if general_parameters['measure_chi_SzSz']:
                        archive['DMFT_results']['it_'+str(it)]['O_tau_'+str(icrsh)] = solvers[icrsh].O_tau

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
            if general_parameters['magnetic']:
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
        if it >= general_parameters['occ_conv_it'] and general_parameters['occ_conv_crit'] > 0.0:
            if mpi.is_master_node():
                # if convergency criteria was already reached dont overwrite it!
                if converged:
                    dummy, std_dev = toolset.check_convergence(sum_k, general_parameters, observables)
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
        elif it == general_parameters['prev_iterations'] + general_parameters['n_iter_dmft']:
            mpi.report('all requested iterations finished')
            mpi.report('#'*80)
            break

        # iteration counter
        it += 1

    return converged, observables, sum_k
