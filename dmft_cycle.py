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
import sys

# triqs
import pytriqs.utility.mpi as mpi
from pytriqs.operators.util.U_matrix import U_matrix, U_matrix_kanamori, reduce_4index_to_2index, U_J_to_radial_integrals
from pytriqs.operators.util.hamiltonians import h_int_kanamori, h_int_slater, h_int_density
from pytriqs.archive import HDFArchive
try:
    # TRIQS 2.0
    from triqs_cthyb.solver import Solver
    from triqs_dft_tools.sumk_dft import SumkDFT
    from pytriqs.gf import GfImTime, GfLegendre, BlockGf, make_hermitian
    from pytriqs.gf.tools import inverse
    from pytriqs.gf.descriptors import Fourier, InverseFourier
    legacy_mode = False
except ImportError:
    # TRIQS 1.4
    from pytriqs.applications.impurity_solvers.cthyb import *
    from pytriqs.applications.dft.sumk_dft import *
    from pytriqs.applications.dft.sumk_dft_tools import *
    from pytriqs.applications.dft.converters.vasp_converter import *
    from pytriqs.applications.dft.converters.plovasp.vaspio import VaspData
    import pytriqs.applications.dft.converters.plovasp.converter as plo_converter
    from pytriqs.gf.local import *
    legacy_mode = True

# own modules
from observables import calc_dft_kin_en, calc_obs, calc_bandcorr_man, write_obs
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
        if not 'DMFT_input'  in archive:
            archive.create_group('DMFT_input')
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

    # if we are not doing a CSC calculation we need to have a DFT mu value
    # if CSC we are using the VASP converter which subtracts mu already
    # if we are at iteration offset 0 we should determine it anyway for safety!
    if (dft_mu == 0.0 and not general_parameters['csc']) or iteration_offset == 0:
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
        toolset.print_block_sym(sum_k, shell_multiplicity)
    else:
        sum_k.read_input_from_hdf(subgrp='DMFT_input', things_to_read=['block_structure', 'rot_mat'])
        sum_k.deg_shells = deg_shells
        toolset.print_block_sym(sum_k, shell_multiplicity)

    # for CSC calculations this does not work
    if general_parameters['magnetic'] and not general_parameters['csc']:
        sum_k.SP = 1
    # if we do AFM calculation we can use symmetry to copy self-energies from
    # one imp to another by exchanging up/down channels for speed up and accuracy
    afm_mapping = []
    if mpi.is_master_node():
        if (general_parameters['magnetic']
                and len(general_parameters['magmom']) == sum_k.n_inequiv_shells
                and general_parameters['afm_order']
                and not 'afm_mapping' in archive['DMFT_input']):

            # find equal or opposite spin imps, where we use the magmom array to
            # identity those with equal numbers or opposite
            # [copy Yes/False, from where, switch up/down channel]
            afm_mapping.append([False, 0, False])

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

                    afm_mapping.append([copy, source, switch])
                else:
                    afm_mapping.append([False, icrsh, False])


            print('AFM calculation selected, mapping self energies as follows:')
            print('imp  [copy sigma, source imp, switch up/down]')
            print('---------------------------------------------')
            for i, elem in enumerate(afm_mapping):
                print(str(i)+' ', elem)
            print('')

            archive['DMFT_input']['afm_mapping'] = afm_mapping

        # else if mapping is already in h5 archive
        elif 'afm_mapping' in archive['DMFT_input']:
            afm_mapping = archive['DMFT_input']['afm_mapping']

        # if anything did not work set afm_order false
        else:
            general_parameters['afm_order'] = False

    general_parameters['afm_order'] = mpi.bcast(general_parameters['afm_order'])
    afm_mapping = mpi.bcast(afm_mapping)

    # build interacting local hamiltonain
    h_int = []
    solvers = []

    # extract U and J
    mpi.report('*** interaction parameters ***')
    for param_name in ('U', 'J'):
        general_parameters = _extract_U_J_list(param_name, sum_k.n_inequiv_shells, general_parameters)

    ## Initialise the Hamiltonian and Solver
    for icrsh in range(sum_k.n_inequiv_shells):
        # ish points to the shell representative of the current group
        ish = sum_k.inequiv_to_corr[icrsh]
        n_orb = sum_k.corr_shells[ish]['dim']
#        l = sum_k.corr_shells[ish]['l']
        orb_names = list(range(n_orb))

        # Construct U matrix of general kanamori type calculations
        if n_orb in (2, 3): # e_g or t_2g cases
            Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=general_parameters['U'][icrsh],
                                            J_hund=general_parameters['J'][icrsh])
        elif n_orb == 5:
            # This is correct when used with the density-density Hamiltonian
            Umat_full = U_matrix(l=2, U_int=general_parameters['U'][icrsh],
                                 J_hund=general_parameters['J'][icrsh], basis='cubic')
            # reduce full 4-index interaction matrix to 2-index
            Umat, Upmat = reduce_4index_to_2index(Umat_full)
            if mpi.is_master_node():
                print('NOTE: The input parameters U and J here are orbital-averaged parameters.')
                print('For the density-density or Slater Hamiltonian (latter not supported yet),')
                print('this corresponds to the definition of U and J in DFT+U, see')
                print('https://cms.mpi.univie.ac.at/wiki/index.php/LDAUTYPE.')

                slater_integrals = U_J_to_radial_integrals(l=2, U_int=general_parameters['U'][icrsh],
                                                           J_hund=general_parameters['J'][icrsh])
                print('The corresponding slater integrals are')
                print('[F0, F2, F4] = [{F[0]:.2f}, {F[1]:.2f}, {F[2]:.2f}]\n'.format(F=slater_integrals))
        else:
            mpi.report('\n*** Hamiltonian for n_orb = {} NOT supported'.format(n_orb))
            sys.exit(2)

        # Construct Hamiltonian
        mpi.report('Constructing the interaction Hamiltonian for shell {}'.format(icrsh))
        if general_parameters['h_int_type'] == 1:
            # 1. density-density
            mpi.report('Using the density-density Hamiltonian ')
            h_int.append(h_int_density(general_parameters['spin_names'], orb_names,
                                       map_operator_structure=sum_k.sumk_to_solver[icrsh],
                                       U=Umat, Uprime=Upmat, H_dump=general_parameters['jobname']+'/'+'H.txt'))
        elif general_parameters['h_int_type'] == 2:
            # 2. Kanamori Hamiltonian
            mpi.report('Using the Kanamori Hamiltonian (with spin-flip and pair-hopping) ')
            h_int.append(h_int_kanamori(general_parameters['spin_names'], orb_names, map_operator_structure=sum_k.sumk_to_solver[icrsh],
                                        off_diag=True, U=Umat, Uprime=Upmat, J_hund=general_parameters['J'][icrsh],
                                        H_dump=general_parameters['jobname']+'/'+'H.txt'))
        elif general_parameters['h_int_type'] == 3:
            # 3. Rotationally-invariant Slater Hamiltonian (4-index)
            raise NotImplementedError('4-index Slater Hamiltonian not implemented.')
            h_int.append(h_int_slater(general_parameters['spin_names'], orb_names_all,
                                      map_operator_structure=map_all,
                                      off_diag=True, U_matrix=Umat_full,
                                      H_dump=general_parameters['jobname']+'/'+'H_full.txt'))

        # save h_int to h5 archive
        if mpi.is_master_node():
            archive['DMFT_input']['h_int'] = h_int

        ####################################
        # hotfix for new triqs 2.0 gf_struct_solver is still a dict
        # but cthyb 2.0 expects a list of pairs ####
        if legacy_mode:
            gf_struct = sum_k.gf_struct_solver[icrsh]
        else:
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

            start_sigma = []
            # load now sigma from other calculation if wanted
            if general_parameters['load_sigma'] and general_parameters['previous_file'] == 'none':
                start_sigma, sum_k.dc_imp, sum_k.dc_energ = toolset.load_sigma_from_h5(general_parameters['path_to_sigma'], general_parameters['load_sigma_iter'])

            # if this is a series of calculation load previous sigma
            elif general_parameters['previous_file'] != 'none':
                start_sigma, sum_k.dc_imp, sum_k.dc_energ = toolset.load_sigma_from_h5(general_parameters['previous_file'], -1)

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

    #############################
    # extract G local
    G_loc_all = sum_k.extract_G_loc()
    #############################

    if mpi.is_master_node():
        # print other system information
        print('\nInverse temperature beta = {}'.format(general_parameters['beta']))
        if solver_parameters['measure_G_l']:
            print('\nSampling G(iw) in Legendre space with {} coefficients'.format(general_parameters['n_LegCoeff']))

    # extract free lattice greens function
    G_loc_all_dft = sum_k.extract_G_loc(with_Sigma=False, mu=dft_mu)
    density_shell_dft = np.zeros(sum_k.n_inequiv_shells)
    density_mat_dft = []
    for icrsh in range(sum_k.n_inequiv_shells):
        density_mat_dft.append([])
        density_mat_dft[icrsh] = G_loc_all_dft[icrsh].density()
        density_shell_dft[icrsh] = G_loc_all_dft[icrsh].total_density()
        mpi.report('total density for imp '+str(icrsh)+' from DFT: '+str(density_shell_dft[icrsh]))


    # extracting new rotation matrices from density_mat or local Hamiltonian
    if ((general_parameters['set_rot'] == 'hloc' or general_parameters['set_rot'] == 'den')
            and iteration_offset == 0 and not general_parameters['load_sigma']):
        if general_parameters['set_rot'] == 'hloc':
            q_diag = sum_k.eff_atomic_levels()
            chnl = 'up'
        elif general_parameters['set_rot'] == 'den':
            q_diag = density_mat_dft
            chnl = 'up_0'

        rot_mat = []
        for icrsh in range(sum_k.n_corr_shells):
            ish = sum_k.corr_to_inequiv[icrsh]
            _, eigvec = np.linalg.eigh(np.real(q_diag[ish][chnl]))
            rot_mat_local = np.array(eigvec) + 0.j

            rot_mat.append(rot_mat_local)

        sum_k.rot_mat = rot_mat
        mpi.report('Updating rotation matrices using dft {} eigenbasis to maximise sign'.format(general_parameters['set_rot']))

        if mpi.is_master_node():
            print('\n new rotation matrices ')
            # rotation matrices
            for icrsh in range(sum_k.n_corr_shells):
                n_orb = sum_k.corr_shells[icrsh]['dim']
                print('rot_mat[{:2d}] '.format(icrsh)+'real part'.center(9*n_orb)+'  '+'imaginary part'.center(9*n_orb))
                rot = np.matrix(sum_k.rot_mat[icrsh])
                for irow in range(n_orb):
                    fmt = '{:9.5f}' * n_orb
                    row = np.real(rot[irow, :]).tolist()[0] + np.imag(rot[irow, :]).tolist()[0]
                    print((' '*11 + fmt + '  ' + fmt).format(*row))

            print('\n')

    if mpi.is_master_node() and iteration_offset == 0:
        # saving rot mat to h5 archive:
        archive['DMFT_input']['rot_mat'] = sum_k.rot_mat

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
        G_loc_all = sum_k.extract_G_loc()

    if ((general_parameters['load_sigma'] or general_parameters['previous_file'] != 'none')
            and iteration_offset == 0):
        sum_k.calc_mu(precision=general_parameters['prec_mu'])
        G_loc_all = sum_k.extract_G_loc()

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

    mpi.report('\n{} DMFT cycles requested. Starting with iteration {}.\n'.format(n_iter, iteration_offset+1))
    # make sure a last time that every node as the same number of iterations
    n_iter = mpi.bcast(n_iter)

    # calculate E_kin_dft for one shot calculations
    if not general_parameters['csc'] and general_parameters['calc_energies']:
        E_kin_dft = calc_dft_kin_en(general_parameters, sum_k, dft_mu)
    else:
        E_kin_dft = None

    converged, observables, sum_k = _dmft_steps(iteration_offset, n_iter, sum_k, solvers, G_loc_all,
                                                general_parameters, solver_parameters, advanced_parameters,
                                                afm_mapping, h_int, archive, shell_multiplicity, E_kin_dft,
                                                observables, dft_mu, G_loc_all_dft, density_mat_dft)

    # for one-shot calculations, we can use the updated GAMMA file for postprocessing
    if not general_parameters['csc'] and general_parameters['oneshot_postproc_gamma_file']:
        # Write the density correction to file after the one-shot calculation
        sum_k.calc_density_correction(filename=os.path.join(general_parameters['jobname'], 'GAMMA'), dm_type='vasp')

    if converged and general_parameters['csc']:
        # is there a smoother way to stop both vasp and triqs from running after convergency is reached?
        if mpi.is_master_node():
            f_stop = open('STOPCAR', 'wt')
            f_stop.write('LABORT = .TRUE.\n')
            f_stop.close()
            del archive
        mpi.MPI.COMM_WORLD.Abort(1)

    mpi.barrier()

    # close the h5 archive
    if mpi.is_master_node():
        if 'archive' in locals():
            del archive

    return observables

def _dmft_steps(iteration_offset, n_iter, sum_k, solvers, G_loc_all, general_parameters, solver_parameters, advanced_parameters,
                afm_mapping, h_int, archive, shell_multiplicity, E_kin_dft, observables, dft_mu, G_loc_all_dft, density_mat_dft):
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

        # looping over inequiv shells and solving for each site seperately
        for icrsh in range(sum_k.n_inequiv_shells):
            # copy the block of G_loc into the corresponding instance of the impurity solver
            solvers[icrsh].G_iw << G_loc_all[icrsh]

            density_shell_pre[icrsh] = solvers[icrsh].G_iw.total_density()
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
            if solver_parameters['measure_G_l']:
                glist_l = []
            for name, g in solvers[icrsh].G0_iw:
                glist_tau.append(GfImTime(indices=g.indices,
                                          beta=general_parameters['beta'],
                                          n_points=solvers[icrsh].n_tau))
                if solver_parameters['measure_G_l']:
                    glist_l.append(GfLegendre(indices=g.indices,
                                              beta=general_parameters['beta'],
                                              n_points=general_parameters['n_LegCoeff']))

            # we will call it G_tau_man for a manual build G_tau
            solvers[icrsh].G_tau_man = BlockGf(name_list=sum_k.gf_struct_solver[icrsh].keys(),
                                               block_list=glist_tau, make_copies=True)
            if solver_parameters['measure_G_l']:
                solvers[icrsh].G_l_man = BlockGf(name_list=sum_k.gf_struct_solver[icrsh].keys(),
                                                 block_list=glist_l, make_copies=True)

            # if we do a AFM calculation we can use the init magnetic moments to
            # copy the self energy instead of solving it explicitly
            if general_parameters['afm_order'] and afm_mapping[icrsh][0]:
                imp_source = afm_mapping[icrsh][1]
                invert_spin = afm_mapping[icrsh][2]
                mpi.report('\ncopying the self-energy for shell {} from shell {}'.format(icrsh, imp_source))
                mpi.report('inverting spin channels: '+str(invert_spin))

                if invert_spin:
                    for spin_channel, _ in sum_k.gf_struct_solver[icrsh].iteritems():
                        if 'up' in spin_channel:
                            target_channel = 'down'+spin_channel.replace('up', '')
                        else:
                            target_channel = 'up'+spin_channel.replace('down', '')

                        solvers[icrsh].Sigma_iw[spin_channel] << solvers[imp_source].Sigma_iw[target_channel]
                        solvers[icrsh].G_tau_man[spin_channel] << solvers[imp_source].G_tau_man[target_channel]
                        solvers[icrsh].G_iw[spin_channel] << solvers[imp_source].G_iw[target_channel]
                        solvers[icrsh].G0_iw[spin_channel] << solvers[imp_source].G0_iw[target_channel]
                        if solver_parameters['measure_G_l']:
                            solvers[icrsh].G_l_man[spin_channel] << solvers[imp_source].G_l_man[target_channel]

                else:
                    solvers[icrsh].Sigma_iw << solvers[imp_source].Sigma_iw
                    solvers[icrsh].G_tau_man << solvers[imp_source].G_tau_man
                    solvers[icrsh].G_iw << solvers[imp_source].G_iw
                    solvers[icrsh].G0_iw << solvers[imp_source].G0_iw
                    if solver_parameters['measure_G_l']:
                        solvers[icrsh].G_l_man << solvers[imp_source].G_l_man

            else:
                # ugly workaround for triqs 1.4
                if legacy_mode:
                    solver_parameters['measure_g_l'] = solver_parameters['measure_G_l']
                    del solver_parameters['measure_G_l']
                    solver_parameters['measure_g_tau'] = solver_parameters['measure_G_tau']
                    del solver_parameters['measure_G_tau']

                ####################################################################
                # Solve the impurity problem for this shell
                mpi.report('\nSolving the impurity problem for shell {} ...'.format(icrsh))
                # *************************************
                solvers[icrsh].solve(h_int=h_int[icrsh], **solver_parameters)
                # *************************************
                ####################################################################

                # revert the changes:
                if legacy_mode:
                    solver_parameters['measure_G_l'] = solver_parameters['measure_g_l']
                    del solver_parameters['measure_g_l']
                    solver_parameters['measure_G_tau'] = solver_parameters['measure_g_tau']
                    del solver_parameters['measure_g_tau']

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
                            solvers[icrsh].G_tau_man[i] << InverseFourier(solvers[icrsh].G_iw[i])
                        # Symmetrize
                        solvers[icrsh].G_iw << make_hermitian(solvers[icrsh].G_iw)
                        # set Sigma and G_iw from G_l
                        solvers[icrsh].Sigma_iw << inverse(solvers[icrsh].G0_iw) - inverse(solvers[icrsh].G_iw)

                        if legacy_mode:
                            # bad ass trick to avoid non asymptotic behavior of Sigma
                            # if legendre is used  with triqs 1.4
                            for key, value in solvers[icrsh].Sigma_iw:
                                solvers[icrsh].Sigma_iw[key].tail[-1] = solvers[icrsh].G_iw[key].tail[-1]

                    # broadcast new G, Sigmas to all other nodes
                    solvers[icrsh].Sigma_iw_orig << mpi.bcast(solvers[icrsh].Sigma_iw_orig)
                    solvers[icrsh].Sigma_iw << mpi.bcast(solvers[icrsh].Sigma_iw)
                    solvers[icrsh].G_iw << mpi.bcast(solvers[icrsh].G_iw)
                    solvers[icrsh].G_tau_man << mpi.bcast(solvers[icrsh].G_tau_man)
                else:
                    solvers[icrsh].G_tau_man << solvers[icrsh].G_tau
                    solvers[icrsh].G_iw << make_hermitian(solvers[icrsh].G_iw)

            # some printout of the obtained density matrices and some basic checks
            density_shell[icrsh] = solvers[icrsh].G_iw.total_density()
            density_tot += density_shell[icrsh]*shell_multiplicity[icrsh]
            density_mat[icrsh] = solvers[icrsh].G_iw.density()
            if mpi.is_master_node():
                print('\nTotal charge of impurity problem: {:7.5f}'.format(density_shell[icrsh]))
                print('Total charge convergency of impurity problem: {:7.5f}'.format(density_shell[icrsh]-density_shell_pre[icrsh]))
                print('\nDensity matrix:')
                for key, value in density_mat[icrsh].iteritems():
                    print(key)
                    print(np.real(value))
                    eige, _ = np.linalg.eigh(value)
                    print('eigenvalues: ', eige)
                    # check for large off-diagonal elements and write out a warning
                    size = len(np.real(value)[0])
                    pr_warning = False
                    for i in range(size):
                        for j in range(size):
                            if i != j and np.real(value)[i, j] >= 0.1:
                                pr_warning = True
                    if pr_warning:
                        print('\n!!! WARNING !!!')
                        print('!!! large off diagonal elements in density matrix detected! I hope you know what you are doing !!!')
                        print('\n!!! WARNING !!!')

        # Done with loop over impurities

        if mpi.is_master_node():
            # Done. Now do post-processing:
            print('\n *** Post-processing the solver output ***')
            print('Total charge of all correlated shells : {:.6f}\n'.format(density_tot))

        # mixing Sigma
        if mpi.is_master_node():
            if it > 1:
                print('mixing sigma with previous iteration by factor '+str(general_parameters['sigma_mix'])+'\n')
                for icrsh in range(sum_k.n_inequiv_shells):
                    solvers[icrsh].Sigma_iw << (general_parameters['sigma_mix'] * solvers[icrsh].Sigma_iw
                                                + (1-general_parameters['sigma_mix']) *  archive['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)])
                    solvers[icrsh].G_iw << (general_parameters['sigma_mix'] * solvers[icrsh].G_iw
                                            + (1-general_parameters['sigma_mix'])*archive['DMFT_results']['last_iter']['Gimp_iw_'+str(icrsh)])

        for icrsh in range(sum_k.n_inequiv_shells):
            solvers[icrsh].Sigma_iw << mpi.bcast(solvers[icrsh].Sigma_iw)
            solvers[icrsh].G_iw << mpi.bcast(solvers[icrsh].G_iw)
        mpi.barrier()

        # calculate new DC
        if general_parameters['dc'] and general_parameters['dc_dmft']:
            sum_k = _calculate_double_counting(sum_k, [solver.G_iw.density() for solver in solvers],
                                               general_parameters, advanced_parameters)

        # symmetrise Sigma
        for icrsh in range(sum_k.n_inequiv_shells):
            sum_k.symm_deg_gf(solvers[icrsh].Sigma_iw, orb=icrsh)

        # doing the dmft loop and set new sigma into sumk
        sum_k.put_Sigma([solvers[icrsh].Sigma_iw for icrsh in range(sum_k.n_inequiv_shells)])

        if general_parameters['fixed_mu']:
            sum_k.set_mu(general_parameters['fixed_mu_value'])
            previous_mu = sum_k.chemical_potential
            mpi.report('+++ Keeping the chemical potential fixed at: '+str(general_parameters['fixed_mu_value'])+' +++')
        else:
            # saving previous mu for writing to observables file
            previous_mu = sum_k.chemical_potential
            sum_k.calc_mu(precision=general_parameters['prec_mu'])

        G_loc_all = sum_k.extract_G_loc()

        # saving results to h5 archive
        if mpi.is_master_node():
            archive['DMFT_results']['iteration_count'] = it
            archive['DMFT_results']['last_iter']['chemical_potential'] = sum_k.chemical_potential
            archive['DMFT_results']['last_iter']['DC_pot'] = sum_k.dc_imp
            archive['DMFT_results']['last_iter']['DC_energ'] = sum_k.dc_energ
            archive['DMFT_results']['last_iter']['dens_mat_pre'] = density_mat_pre
            archive['DMFT_results']['last_iter']['dens_mat_post'] = density_mat
            for icrsh in range(sum_k.n_inequiv_shells):
                archive['DMFT_results']['last_iter']['G0_iw'+str(icrsh)] = solvers[icrsh].G0_iw
                archive['DMFT_results']['last_iter']['Gimp_tau_'+str(icrsh)] = solvers[icrsh].G_tau_man
                if solver_parameters['measure_G_l']:
                    archive['DMFT_results']['last_iter']['Gimp_l_'+str(icrsh)] = solvers[icrsh].G_l_man
                archive['DMFT_results']['last_iter']['Gimp_iw_'+str(icrsh)] = solvers[icrsh].G_iw
                archive['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)] = solvers[icrsh].Sigma_iw

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
                    archive['DMFT_results']['it_'+str(it)]['Gimp_tau_'+str(icrsh)] = solvers[icrsh].G_tau_man
                    if solver_parameters['measure_G_l']:
                        archive['DMFT_results']['it_'+str(it)]['Gimp_l_'+str(icrsh)] = solvers[icrsh].G_l_man
                    archive['DMFT_results']['it_'+str(it)]['Gimp_iw_'+str(icrsh)] = solvers[icrsh].G_iw
                    archive['DMFT_results']['it_'+str(it)]['Sigma_iw_'+str(icrsh)] = solvers[icrsh].Sigma_iw

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
            observables = calc_obs(observables,
                                   general_parameters,
                                   solver_parameters,
                                   it,
                                   solvers,
                                   h_int,
                                   dft_mu,
                                   previous_mu,
                                   sum_k,
                                   G_loc_all_dft,
                                   density_mat_dft,
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

        # check also if the global number of iterations is reached
        if it == general_parameters['n_iter_dmft']:
            mpi.report('all requested iterations finished')
            mpi.report('#'*80)
            break

        # iteration counter
        it += 1

    return converged, observables, sum_k
