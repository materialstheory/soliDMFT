# defines the dmft_cycle which works for one-shot and csc equally

# the future numpy (>1.15) is not fully compatible with triqs 2.0 atm
# suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# system
import numpy as np

# triqs
import pytriqs.utility.mpi as mpi
from pytriqs.operators.util.U_matrix import U_matrix, U_matrix_kanamori, reduce_4index_to_2index
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

def dmft_cycle(general_parameters, solver_parameters, observables):
    """
    main dmft cycle that works for one shot and CSC equally

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    solver_parameters : dict
        solver parameters as a dict
    observables : dict
        current observable array for calculation

    __Returns:__
    observables : dict
        updated observable array for calculation
    """

    # create Sumk object
    if general_parameters['csc']:
        SK = SumkDFT(hdf_file = general_parameters['seedname']+'.h5',use_dft_blocks = False,h_field=general_parameters['h_field'])
    else:
        SK = SumkDFT(hdf_file = general_parameters['jobname']+'/'+general_parameters['seedname']+'.h5',use_dft_blocks = False,h_field=general_parameters['h_field'])

    iteration_offset = 0
    dft_mu = 0.0

    # determine chemical potential for bare DFT SK object
    if mpi.is_master_node():
        ar = HDFArchive(general_parameters['jobname']+'/'+general_parameters['seedname']+'.h5','a')
        if not 'DMFT_results' in ar: ar.create_group('DMFT_results')
        if not 'last_iter' in ar['DMFT_results']: ar['DMFT_results'].create_group('last_iter')
        if not 'DMFT_input'  in ar: ar.create_group('DMFT_input')
        if 'iteration_count' in ar['DMFT_results']:
            iteration_offset = ar['DMFT_results']['iteration_count']
            SK.chemical_potential = ar['DMFT_results']['last_iter']['chemical_potential']
        if general_parameters['dft_mu'] != 0.0:
            dft_mu = general_parameters['dft_mu']
    # cast everything to other nodes
    SK.chemical_potential = mpi.bcast(SK.chemical_potential)
    dft_mu = mpi.bcast(dft_mu)

    iteration_offset = mpi.bcast(iteration_offset)

    if iteration_offset == 0 and dft_mu != 0.0:
        SK.chemical_potential = dft_mu
        mpi.report("\n chemical potential set to "+str(SK.chemical_potential)+" eV \n")


    # determine block structure for solver
    det_blocks = True
    shell_multiplicity = []
    deg_shells = []
    # load previous block_structure if possible
    if mpi.is_master_node():
        if 'block_structure' in ar['DMFT_input']:
            det_blocks = False
            shell_multiplicity = ar['DMFT_input']['shell_multiplicity']
            deg_shells = ar['DMFT_input']['deg_shells']
    det_blocks = mpi.bcast(det_blocks)
    deg_shells = mpi.bcast(deg_shells)
    shell_multiplicity = mpi.bcast(shell_multiplicity)

    # if we are not doing a CSC calculation we need to have a DFT mu value
    # if CSC we are using the VASP converter which subtracts mu already
    # if we are at iteration offset 0 we should determine it anyway for safety!
    if (dft_mu == 0.0 and not general_parameters['csc']) or iteration_offset == 0:
        dft_mu = SK.calc_mu( precision = general_parameters['prec_mu'] )

    # determine block structure for GF and Hyb function
    if det_blocks and not general_parameters['load_sigma']:
        SK, shell_multiplicity = toolset.determine_block_structure(SK, general_parameters)
    # if load sigma we need to load everything from this h5 archive
    elif general_parameters['load_sigma']:
        deg_shells = []
        # loading shell_multiplicity
        if mpi.is_master_node():
            old_calc = HDFArchive(general_parameters['path_to_sigma'],'r')
            shell_multiplicity = old_calc['DMFT_input']['shell_multiplicity']
            deg_shells = old_calc['DMFT_input']['deg_shells']
            del old_calc
        shell_multiplicity = mpi.bcast(shell_multiplicity)
        deg_shells = mpi.bcast(deg_shells)
        #loading block_struc and rot mat
        SK_old = SumkDFT(hdf_file = general_parameters['path_to_sigma'])
        SK_old.read_input_from_hdf(subgrp='DMFT_input',things_to_read=['block_structure','rot_mat'])
        SK.block_structure = SK_old.block_structure
        if general_parameters['magnetic']:
            SK.deg_shells = [[] for icrsh in range(0,SK.n_inequiv_shells)]
        else:
            SK.deg_shells = deg_shells
        SK.rot_mat = SK_old.rot_mat
        toolset.print_block_sym(SK, shell_multiplicity)
    else:
        SK.read_input_from_hdf(subgrp='DMFT_input',things_to_read=['block_structure','rot_mat'])
        SK.deg_shells = deg_shells
        toolset.print_block_sym(SK, shell_multiplicity)

    # for CSC calculations this does not work
    if general_parameters['magnetic'] and not general_parameters['csc']:
        SK.SP = 1
    # if we do AFM calculation we can use symmetry to copy self-energies from
    # one imp to another by exchanging up/down channels for speed up and accuracy
    afm_mapping = []
    if mpi.is_master_node():
        if (general_parameters['magnetic'] and
            len(general_parameters['magmom']) == SK.n_inequiv_shells and
            general_parameters['afm_order'] and
            not 'afm_mapping' in ar['DMFT_input']):

            # find equal or opposite spin imps, where we use the magmom array to
            # identity those with equal numbers or opposite
            # [copy Yes/False, from where, switch up/down channel]
            afm_mapping.append([False,0,False])

            abs_moms = map(abs,general_parameters['magmom'])

            for icrsh in range(1,SK.n_inequiv_shells):
                # if the moment was seen before ...
                if abs_moms[icrsh] in abs_moms[0:icrsh]:
                    copy = True
                    # find the source imp to copy from
                    source = abs_moms[0:icrsh].index(abs_moms[icrsh])

                    # determine if we need to switch up and down channel
                    if general_parameters['magmom'][icrsh] == general_parameters['magmom'][source]:
                        switch = False
                    elif general_parameters['magmom'][icrsh] == -1*general_parameters['magmom'][source]:
                        switch = True
                    # double check if the moment where not the same and then don't copy
                    else:
                        switch = False
                        copy = False

                    afm_mapping.append([copy,source,switch])
                else:
                    afm_mapping.append([False,icrsh,False])


            print('AFM calculation selected, mapping self energies as follows:')
            print('imp  [copy sigma, source imp, switch up/down]')
            print('---------------------------------------------')
            for i,elem in enumerate(afm_mapping):
                print(str(i)+' ', elem)
            print('')

            ar['DMFT_input']['afm_mapping'] = afm_mapping

        # else if mapping is already in h5 archive
        elif 'afm_mapping' in ar['DMFT_input']:
            afm_mapping = ar['DMFT_input']['afm_mapping']

        # if anything did not work set afm_order false
        else:
            general_parameters['afm_order'] = False

    general_parameters['afm_order'] = mpi.bcast(general_parameters['afm_order'])
    afm_mapping = mpi.bcast(afm_mapping)

    # build interacting local hamiltonain
    h_int = []
    S = []
    # extract U and J
    mpi.report('*** interaction parameters ***')
    if len(general_parameters['U']) == 1:
        mpi.report("Assuming %s=%.2f for all correlated shells"%('U',general_parameters['U'][0]))
        general_parameters['U'] = general_parameters['U']*SK.n_inequiv_shells
    elif len(general_parameters['U']) == SK.n_inequiv_shells:
        mpi.report('U list for correlated shells: '+str(general_parameters['U']))
    else:
        raise IndexError("Property list %s must have length 1 or n_inequiv_shells"%(str(general_parameters['U'])))

    # now J
    if len(general_parameters['J']) == 1:
        mpi.report("Assuming %s=%.2f for all correlated shells"%('J',general_parameters['J'][0]))
        general_parameters['J'] = general_parameters['J']*SK.n_inequiv_shells
    elif len(general_parameters['J']) == SK.n_inequiv_shells:
        mpi.report('J list for correlated shells: '+str(general_parameters['J']))
    else:
        raise IndexError("Property list %s must have length 1 or n_inequiv_shells"%(str(general_parameters['J'])))

    ## Initialise the Hamiltonian and Solver
    for icrsh in range(SK.n_inequiv_shells):
        # ish points to the shell representative of the current group
        ish = SK.inequiv_to_corr[icrsh]
        n_orb = SK.corr_shells[ish]['dim']
        l = SK.corr_shells[ish]['l']
        orb_names = list(range(n_orb))

        # Construct U matrix of general kanamori type calculations
        if n_orb == 2 or n_orb == 3: # e_g or t_2g cases
            Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=general_parameters['U'][icrsh], J_hund=general_parameters['J'][icrsh])
        elif n_orb == 5:
            # This is correct when used with the density-density Hamiltonian
            # Calculation of F0, F2, F4 from doi.org/10.1103/PhysRevB.90.165105
            R = 0.63 # SumkDFT value for F4/F2
            F2 = 49.0 / (3.0 + 20.0/9.0 * R) * general_parameters['J'][icrsh]
            F4 = R * F2
            F0 = general_parameters['U'][icrsh] - 4./49. * (F2 + F4)

            Umat_full = U_matrix(l=2, radial_integrals=[F0, F2, F4], basis='cubic')
            # reduce full 4-index interaction matrix to 2-index
            Umat, Upmat = reduce_4index_to_2index(Umat_full)
            if mpi.is_master_node():
                Uavg = F0
                Javg = (F2 + F4)/14.0
                print('\nUav =%9.4f, Jav =%9.4f, F4/F2 =%9.4f'%(Uavg,Javg,R))
                print('F0  =%9.4f, F2  =%9.4f, F4    =%9.4f'%(F0,F2,F4))
        else:
            mpi.report( '\n*** Hamiltonian for n_orb = %s NOT supported'%(n_orb) )
            quit()

        # Construct Hamiltonian
        mpi.report('Constructing the interaction Hamiltonian for shell %s '%(icrsh))
        if general_parameters['h_int_type'] == 1:
            # 1. density-density
            mpi.report('Using the density-density Hamiltonian ')
            h_int.append(h_int_density(general_parameters['spin_names'], orb_names, map_operator_structure=SK.sumk_to_solver[icrsh],
                        U=Umat, Uprime=Upmat, H_dump=general_parameters['jobname']+'/'+"H.txt") )
        elif general_parameters['h_int_type'] == 2:
            # 2. Kanamori Hamiltonian
            mpi.report('Using the Kanamori Hamiltonian (with spin-flip and pair-hopping) ')
            h_int.append( h_int_kanamori(general_parameters['spin_names'], orb_names, map_operator_structure=SK.sumk_to_solver[icrsh],
                          off_diag=True, U=Umat, Uprime=Upmat, J_hund=general_parameters['J'][icrsh], H_dump=general_parameters['jobname']+'/'+"H.txt") )
        elif general_parameters['h_int_type'] == 3:
            # 3. Rotationally-invariant Slater Hamiltonian (4-index)
            h_int.append( h_int_slater(general_parameters['spin_names'], orb_names_all, map_operator_structure=map_all,
                    off_diag=True, U_matrix=Umat_full, H_dump=general_parameters['jobname']+'/'+"H_full.txt") )

        # save h_int to h5 archive
        if mpi.is_master_node():
            ar['DMFT_input']['h_int'] = h_int

        ####################################
        # hotfix for new triqs 2.0 gf_struct_solver is still a dict
        # but cthyb 2.0 expects a list of pairs ####
        if legacy_mode:
            gf_struct = SK.gf_struct_solver[icrsh]
        else:
            gf_struct = [ [k, v] for k, v in SK.gf_struct_solver[icrsh].iteritems() ]
        ####################################
        # Construct the Solver instances
        if solver_parameters["measure_G_l"]:
            S.append( Solver(beta=general_parameters['beta'], gf_struct=gf_struct, n_l=general_parameters["n_LegCoeff"]) )
        else :
            S.append( Solver(beta=general_parameters['beta'], gf_struct=gf_struct) )

    # if Sigma is loaded, mu needs to be calculated again
    calc_mu = False

    # Prepare hdf file and and check for previous iterations
    if mpi.is_master_node():
        obs_prev = []
        if 'iteration_count' in ar['DMFT_results']:
            print('\n *** loading previous self energies ***')
            SK.dc_imp = ar['DMFT_results']['last_iter']['DC_pot']
            SK.dc_energ = ar['DMFT_results']['last_iter']['DC_energ']
            if 'observables' in ar['DMFT_results']:
                obs_prev = ar['DMFT_results']['observables']
            for icrsh in range(SK.n_inequiv_shells):
                print('loading Sigma_imp'+str(icrsh)+' from previous calculation')
                S[icrsh].Sigma_iw = ar['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)]
            calc_mu = True
        else:
            # calculation from scratch:
            ## write some input parameters to the archive
            ar['DMFT_input']['general_parameters'] = general_parameters
            ar['DMFT_input']['solver_parameters'] = solver_parameters
            ## and also the SumK <--> Solver mapping (used for restarting)
            for item in ['block_structure','deg_shells']: ar['DMFT_input'][item] = getattr(SK, item)
            # and the shell_multiplicity
            ar['DMFT_input']['shell_multiplicity'] = shell_multiplicity

            start_sigma = []
            # load now sigma from other calculation if wanted
            if general_parameters['load_sigma'] == True and general_parameters['previous_file'] == 'none':
                start_sigma, SK.dc_imp, SK.dc_energ = toolset.load_sigma_from_h5(general_parameters['path_to_sigma'], general_parameters['load_sigma_iter'])

            # if this is a series of calculation load previous sigma
            elif general_parameters['previous_file'] != 'none':
                start_sigma, SK.dc_imp, SK.dc_energ= toolset.load_sigma_from_h5(general_parameters['previous_file'],-1)

            # load everything now to the solver
            if start_sigma:
                calc_mu = True
                for icrsh in range(SK.n_inequiv_shells):
                    S[icrsh].Sigma_iw = start_sigma[icrsh]

    # bcast everything to other nodes
    for icrsh in range(SK.n_inequiv_shells):
        S[icrsh].Sigma_iw = mpi.bcast(S[icrsh].Sigma_iw)
        S[icrsh].G_iw = mpi.bcast(S[icrsh].G_iw)
    SK.dc_imp = mpi.bcast(SK.dc_imp)
    SK.dc_energ = mpi.bcast(SK.dc_energ)
    SK.set_dc(SK.dc_imp,SK.dc_energ)
    calc_mu = mpi.bcast(calc_mu)


    # symmetrise Sigma
    for icrsh in range(SK.n_inequiv_shells): SK.symm_deg_gf(S[icrsh].Sigma_iw,orb=icrsh)

    SK.put_Sigma([S[icrsh].Sigma_iw for icrsh in range(SK.n_inequiv_shells)])
    if calc_mu:
        # determine chemical potential
        SK.calc_mu( precision = general_parameters['prec_mu'] )

    #############################
    # extract G local
    G_loc_all = SK.extract_G_loc()
    #############################

    if general_parameters['occ_conv_crit'] > 0.0:
        conv_file = open(general_parameters['jobname']+'/'+'convergence.dat', 'a')

    if mpi.is_master_node():
        # print other system information
        print("\nInverse temperature beta = %s"%(general_parameters['beta']))
        if solver_parameters["measure_G_l"]:
            print("\nSampling G(iw) in Legendre space with %s coefficients"%(general_parameters["n_LegCoeff"]))

    # extract free lattice greens function
    G_loc_all_dft = SK.extract_G_loc(with_Sigma=False,mu=dft_mu)
    density_shell_dft = np.zeros(SK.n_inequiv_shells)
    density_mat_dft = []
    for icrsh in range(SK.n_inequiv_shells):
        density_mat_dft.append([])
        density_mat_dft[icrsh] = G_loc_all_dft[icrsh].density()
        density_shell_dft[icrsh] = G_loc_all_dft[icrsh].total_density()
        mpi.report('total density for imp '+str(icrsh)+' from DFT: '+str(density_shell_dft[icrsh]))


    # extracting new rotation matrices from density_mat or local Hamiltonian
    if (general_parameters['set_rot'] == 'hloc' or general_parameters['set_rot'] == 'den') and iteration_offset == 0 and general_parameters['load_sigma'] == False:
        if general_parameters['set_rot'] == 'hloc':
            q_diag = SK.eff_atomic_levels()
            chnl = 'up'
        elif general_parameters['set_rot'] == 'den':
            q_diag = density_mat_dft
            chnl = 'up_0'

        rot_mat = []
        for icrsh in range(SK.n_corr_shells):
            ish = SK.corr_to_inequiv[icrsh]
            eigval, eigvec = np.linalg.eigh(np.real(q_diag[ish][chnl]))
            rot_mat_local = np.array(eigvec) + 0.j

            rot_mat.append(rot_mat_local)

        SK.rot_mat = rot_mat
        mpi.report('Updating rotation matrices using dft %s eigenbasis to maximise sign'%(general_parameters['set_rot']))

        if mpi.is_master_node():
            print("\n new rotation matrices ")
            # rotation matrices
            for icrsh in range(SK.n_corr_shells):
                n_orb = SK.corr_shells[icrsh]['dim']
                print('rot_mat[%2d] '%(icrsh)+'real part'.center(9*n_orb)+'  '+'imaginary part'.center(9*n_orb))
                rot = np.matrix( SK.rot_mat[icrsh] )
                for irow in range(n_orb):
                    fmt = '{:9.5f}' * n_orb
                    row = np.real(rot[irow,:]).tolist()[0] + np.imag(rot[irow,:]).tolist()[0]
                    print(('           '+fmt+'  '+fmt).format(*row))

            print('\n')

    if mpi.is_master_node() and iteration_offset == 0:
        # saving rot mat to h5 archive:
        ar['DMFT_input']['rot_mat'] = SK.rot_mat

    #Double Counting if first iteration or if CSC calculation with DFTDC
    if ((iteration_offset == 0 and general_parameters['dc'] and not general_parameters['load_sigma']) or
       (general_parameters['csc'] and general_parameters['dc'] and not general_parameters['dc_dmft'])):
        mpi.report('\n *** DC determination ***')

        for icrsh in range(SK.n_inequiv_shells):
            ###################################################################
            if general_parameters['dc_type'] == 3:
                # this is FLL for eg orbitals only as done in Seth PRB 96 205139 2017 eq 10
                # this setting for U and J is reasonable as it is in the spirit of F0 and Javg
                # for the 5 orb case
                mpi.report('Doing FLL DC for eg orbitals only with Uavg=U-J and Javg=2*J')
                Uavg = general_parameters['U'][icrsh] - general_parameters['J'][icrsh]
                Javg = 2*general_parameters['J'][icrsh]
                SK.calc_dc(density_mat_dft[icrsh], U_interact = Uavg, J_hund = Javg, orb = icrsh, use_dc_formula = 0)
            else:
                SK.calc_dc(density_mat_dft[icrsh], U_interact = general_parameters['U'][icrsh], J_hund = general_parameters['J'][icrsh], orb = icrsh, use_dc_formula = general_parameters['dc_type'])

            ###################################################################

    # initialise sigma if first iteration
    if (iteration_offset == 0 and
        general_parameters['previous_file'] == 'none' and
        general_parameters['load_sigma'] == False and
        general_parameters['dc']):
        for icrsh in range(SK.n_inequiv_shells):
            # if we are doing a mangetic calculation and initial magnetic moments
            # are set, manipulate the initial sigma accordingly
            if general_parameters['magnetic'] and general_parameters['magmom']:
                fac = abs(general_parameters['magmom'][icrsh])

                # init self energy according to factors in magmoms
                if general_parameters['magmom'][icrsh] > 0.0:
                    # if larger 1 the up channel will be favored
                    for spin_channel, elem in SK.gf_struct_solver[icrsh].iteritems():
                        if 'up' in spin_channel:
                            S[icrsh].Sigma_iw[spin_channel] << (1+fac)*SK.dc_imp[SK.inequiv_to_corr[icrsh]]['up'][0,0]
                        else:
                            S[icrsh].Sigma_iw[spin_channel] <<  (1-fac)*SK.dc_imp[SK.inequiv_to_corr[icrsh]]['down'][0,0]
                else:
                    for spin_channel, elem in SK.gf_struct_solver[icrsh].iteritems():
                        if 'down' in spin_channel:
                            S[icrsh].Sigma_iw[spin_channel] << (1+fac)*SK.dc_imp[SK.inequiv_to_corr[icrsh]]['up'][0,0]
                        else:
                            S[icrsh].Sigma_iw[spin_channel] <<  (1-fac)*SK.dc_imp[SK.inequiv_to_corr[icrsh]]['down'][0,0]
            else:
                S[icrsh].Sigma_iw << SK.dc_imp[SK.inequiv_to_corr[icrsh]]['up'][0,0]

        # set DC as Sigma and extract the new Gloc with DC potential
        SK.put_Sigma([S[icrsh].Sigma_iw for icrsh in range(SK.n_inequiv_shells)])
        G_loc_all = SK.extract_G_loc()

    if ((general_parameters['load_sigma'] == True or general_parameters['previous_file'] != 'none')
        and iteration_offset == 0):
        SK.calc_mu( precision = general_parameters['prec_mu'] )
        G_loc_all = SK.extract_G_loc()

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

    mpi.report('\n%s DMFT cycles requested. Starting with iteration %s. \n'%(n_iter,iteration_offset+1))
    # make sure a last time that every node as the same number of iterations
    n_iter = mpi.bcast(n_iter)

    # calculate E_kin_dft for one shot calculations
    if not general_parameters['csc'] and general_parameters['calc_energies']:
        E_kin_dft = calc_dft_kin_en(general_parameters, SK, dft_mu)

    sampling = False
    dmft_looping = True
    it = iteration_offset+1
    # The infamous DMFT self consistency cycle
    while (dmft_looping == True):
        mpi.report("#"*80)
        mpi.report('Running iteration: '+str(it)+' / '+str(iteration_offset+n_iter))

        # init local density matrices for observables
        density_tot = 0.0
        density_shell = np.zeros(SK.n_inequiv_shells)
        density_mat  = []
        for icrsh in range(SK.n_inequiv_shells):
            density_mat.append([])
        density_shell_pre = np.zeros(SK.n_inequiv_shells)
        density_mat_pre  = []
        for icrsh in range(SK.n_inequiv_shells):
            density_mat_pre.append([])

        mpi.barrier()
        for icrsh in range(SK.n_inequiv_shells): SK.symm_deg_gf(S[icrsh].Sigma_iw,orb=icrsh)

        # looping over inequiv shells and solving for each site seperately
        for icrsh in range(SK.n_inequiv_shells):
            # copy the block of G_loc into the corresponding instance of the impurity solver
            S[icrsh].G_iw << G_loc_all[icrsh]

            density_shell_pre[icrsh] = S[icrsh].G_iw.total_density()
            mpi.report( "\n *** Correlated Shell type #%3d : "%(icrsh) + "Total charge of impurity problem = %.6f"%(density_shell_pre[icrsh]))
            density_mat_pre[icrsh] = S[icrsh].G_iw.density()
            mpi.report( "Density matrix:")
            for key,value in density_mat_pre[icrsh].iteritems():
                mpi.report( key )
                mpi.report( np.real(value))

            # dyson equation to extract G0_iw
            S[icrsh].G0_iw << inverse(S[icrsh].Sigma_iw + inverse(S[icrsh].G_iw))

            # impose fundamental symmetries
            S[icrsh].G0_iw << make_hermitian(S[icrsh].G0_iw)

            SK.symm_deg_gf(S[icrsh].G0_iw,orb=icrsh)

            # prepare our G_tau and G_l used to save the 'good' G_tau
            glist_tau = []
            if solver_parameters["measure_G_l"]:
                glist_l = []
            for name, g in S[icrsh].G0_iw:
                glist_tau.append(GfImTime(indices =g.indices,
                                          beta = general_parameters['beta'],
                                          n_points= S[icrsh].n_tau)
                                          )
                if solver_parameters["measure_G_l"]:
                    glist_l.append(GfLegendre(indices =g.indices,
                                              beta = general_parameters['beta'],
                                              n_points= general_parameters['n_LegCoeff'])
                                              )

            # we will call it G_tau_man for a manual build G_tau
            S[icrsh].G_tau_man = BlockGf(name_list = SK.gf_struct_solver[icrsh].keys(),
                                         block_list = glist_tau,
                                         make_copies = True)
            if solver_parameters["measure_G_l"]:
                S[icrsh].G_l_man = BlockGf(name_list = SK.gf_struct_solver[icrsh].keys(),
                                           block_list = glist_l,
                                           make_copies = True)

            # if we do a AFM calculation we can use the init magnetic moments to
            # copy the self energy instead of solving it explicitly
            if general_parameters['afm_order'] and afm_mapping[icrsh][0] == True:
                imp_source = afm_mapping[icrsh][1]
                invert_spin = afm_mapping[icrsh][2]
                mpi.report("\ncopying the self-energy for shell %d from shell %d"%(icrsh,imp_source))
                mpi.report("inverting spin channels: "+str(invert_spin))

                if invert_spin:
                    for spin_channel, elem in SK.gf_struct_solver[icrsh].iteritems():
                        if 'up' in spin_channel:
                            target_channel = 'down'+spin_channel.replace('up','')
                        else:
                            target_channel = 'up'+spin_channel.replace('down','')

                        S[icrsh].Sigma_iw[spin_channel] << S[imp_source].Sigma_iw[target_channel]
                        S[icrsh].G_tau_man[spin_channel] << S[imp_source].G_tau_man[target_channel]
                        S[icrsh].G_iw[spin_channel] << S[imp_source].G_iw[target_channel]
                        S[icrsh].G0_iw[spin_channel] << S[imp_source].G0_iw[target_channel]
                        if solver_parameters["measure_G_l"]:
                            S[icrsh].G_l_man[spin_channel] << S[imp_source].G_l_man[target_channel]

                else:
                    S[icrsh].Sigma_iw << S[imp_source].Sigma_iw
                    S[icrsh].G_tau_man << S[imp_source].G_tau_man
                    S[icrsh].G_iw << S[imp_source].G_iw
                    S[icrsh].G0_iw << S[imp_source].G0_iw
                    if solver_parameters["measure_G_l"]:
                        S[icrsh].G_l_man << S[imp_source].G_l_man

            else:
                # ugly workaround for triqs 1.4
                if legacy_mode:
                    solver_parameters["measure_g_l"] = solver_parameters["measure_G_l"]
                    del solver_parameters["measure_G_l"]
                    solver_parameters["measure_g_tau"] = solver_parameters["measure_G_tau"]
                    del solver_parameters["measure_G_tau"]

                ####################################################################
                # Solve the impurity problem for this shell
                mpi.report("\nSolving the impurity problem for shell %d ..."%(icrsh))
                # *************************************
                S[icrsh].solve(h_int=h_int[icrsh], **solver_parameters)
                # *************************************
                ####################################################################

                # revert the changes:
                if legacy_mode:
                    solver_parameters["measure_G_l"] = solver_parameters["measure_g_l"]
                    del solver_parameters["measure_g_l"]
                    solver_parameters["measure_G_tau"] = solver_parameters["measure_g_tau"]
                    del solver_parameters["measure_g_tau"]

                # use Legendre for next G and Sigma instead of matsubara, less noisy!
                if solver_parameters["measure_G_l"]:
                    S[icrsh].Sigma_iw_orig = S[icrsh].Sigma_iw.copy()
                    S[icrsh].G_iw_from_leg = S[icrsh].G_iw.copy()
                    S[icrsh].G_l_man << S[icrsh].G_l
                    if mpi.is_master_node():
                        for i, g in S[icrsh].G_l:
                            g.enforce_discontinuity(np.identity(g.target_shape[0]))
                            S[icrsh].G_iw[i].set_from_legendre(g)
                            # update G_tau as well:
                            S[icrsh].G_tau_man[i] << InverseFourier(S[icrsh].G_iw[i])
                        # Symmetrize
                        S[icrsh].G_iw << make_hermitian(S[icrsh].G_iw)

                        # set Sigma and G_iw from G_l
                        S[icrsh].Sigma_iw << inverse(S[icrsh].G0_iw) - inverse(S[icrsh].G_iw)

                        if legacy_mode:
                            # bad ass trick to avoid non asymptotic behavior of Sigma
                            # if legendre is used  with triqs 1.4
                            for key, value in S[icrsh].Sigma_iw:
                                S[icrsh].Sigma_iw[key].tail[-1] = S[icrsh].G_iw[key].tail[-1]

                    # broadcast new G, Sigmas to all other nodes
                    S[icrsh].Sigma_iw_orig << mpi.bcast(S[icrsh].Sigma_iw_orig)
                    S[icrsh].Sigma_iw << mpi.bcast(S[icrsh].Sigma_iw)
                    S[icrsh].G_iw << mpi.bcast(S[icrsh].G_iw)
                    S[icrsh].G_tau_man << mpi.bcast(S[icrsh].G_tau_man)
                else:
                    S[icrsh].G_tau_man << S[icrsh].G_tau
                    S[icrsh].G_iw << make_hermitian(S[icrsh].G_iw)

            # some printout of the obtained density matrices and some basic checks
            density_shell[icrsh] = S[icrsh].G_iw.total_density()
            density_tot += density_shell[icrsh]*shell_multiplicity[icrsh]
            density_mat[icrsh] = S[icrsh].G_iw.density()
            if mpi.is_master_node():
                print("\nTotal charge of impurity problem : "+"{:7.5f}".format(density_shell[icrsh]))
                print("Total charge convergency of impurity problem : "+"{:7.5f}".format(density_shell[icrsh]-density_shell_pre[icrsh]))
                print("\nDensity matrix:")
                for key,value in density_mat[icrsh].iteritems():
                    print(key)
                    print(np.real(value))
                    eige, eigv = np.linalg.eigh(value)
                    print('eigenvalues: ', eige)
                    # check for large off-diagonal elements and write out a warning
                    i = 0
                    j = 0
                    size = len(np.real(value)[0,:])
                    pr_warning = False
                    for i in range(0,size):
                        for j in range(0,size):
                            if i!=j and np.real(value)[i,j] >= 0.1:
                                pr_warning = True
                    if pr_warning:
                        print('\n!!! WARNING !!!')
                        print('!!! large off diagonal elements in density matrix detected! I hope you know what you are doing !!!')
                        print('\n!!! WARNING !!!')

        # Done with loop over impurities

        if mpi.is_master_node():
            # Done. Now do post-processing:
            print("\n *** Post-processing the solver output ***")
            print("Total charge of all correlated shells : %.6f \n"%density_tot)

        # mixing Sigma
        if mpi.is_master_node():
            if it > 1:
                print("mixing sigma with previous iteration by factor "+str(general_parameters['sigma_mix'])+'\n')
                for icrsh in range(SK.n_inequiv_shells):
                    S[icrsh].Sigma_iw << ( general_parameters['sigma_mix'] * S[icrsh].Sigma_iw
                                        +(1-general_parameters['sigma_mix']) *  ar['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)] )
                    S[icrsh].G_iw << ( general_parameters['sigma_mix'] * S[icrsh].G_iw
                                    +(1-general_parameters['sigma_mix'])*ar['DMFT_results']['last_iter']['Gimp_iw_'+str(icrsh)] )

        for icrsh in range(SK.n_inequiv_shells):
            S[icrsh].Sigma_iw << mpi.bcast(S[icrsh].Sigma_iw)
            S[icrsh].G_iw << mpi.bcast(S[icrsh].G_iw)
        mpi.barrier()

        # calculate new DC
        if general_parameters['dc_dmft'] and general_parameters['dc']:
            for icrsh in range(SK.n_inequiv_shells):
                dm = S[icrsh].G_iw.density()
                if general_parameters['dc_type'] == 3:
                    # this is FLL for eg orbitals only as done in Seth PRB 96 205139 2017 eq 10
                    # this setting for U and J is reasonable as it is in the spirit of F0 and Javg
                    # for the 5 orb case
                    mpi.report('Doing FLL DC for eg orbitals only with Uavg=U-J and Javg=2*J')
                    Uavg = general_parameters['U'][icrsh] - general_parameters['J'][icrsh]
                    Javg = 2*general_parameters['J'][icrsh]
                    SK.calc_dc(dm, U_interact = Uavg, J_hund = Javg, orb = icrsh, use_dc_formula = 0)
                else:
                    SK.calc_dc(dm, U_interact = general_parameters['U'][icrsh], J_hund = general_parameters['J'][icrsh], orb = icrsh, use_dc_formula = general_parameters['dc_type'])

        # symmetrise Sigma
        for icrsh in range(SK.n_inequiv_shells): SK.symm_deg_gf(S[icrsh].Sigma_iw,orb=icrsh)

        # doing the dmft loop and set new sigma into sumk
        SK.put_Sigma([S[icrsh].Sigma_iw for icrsh in range(SK.n_inequiv_shells)])

        if general_parameters['fixed_mu']:
            SK.set_mu(general_parameters['fixed_mu_value'])
            previous_mu = SK.chemical_potential
            mpi.report( "+++ Keeping the  chemical potential fixed  at: "+str(general_parameters['fixed_mu_value'])+" +++ " )
        else:
            # saving previous mu for writing to observables file
            previous_mu = SK.chemical_potential
            SK.calc_mu( precision = general_parameters['prec_mu'] )

        # extract new G_loc
        G_loc_all = SK.extract_G_loc()


        # saving results to h5 archive
        if mpi.is_master_node():
            ar['DMFT_results']['iteration_count'] = it
            ar['DMFT_results']['last_iter']['chemical_potential'] = SK.chemical_potential
            ar['DMFT_results']['last_iter']['DC_pot'] = SK.dc_imp
            ar['DMFT_results']['last_iter']['DC_energ'] = SK.dc_energ
            ar['DMFT_results']['last_iter']['dens_mat_pre'] = density_mat_pre
            ar['DMFT_results']['last_iter']['dens_mat_post'] = density_mat
            for icrsh in range(SK.n_inequiv_shells):
                ar['DMFT_results']['last_iter']['G0_iw'+str(icrsh)] = S[icrsh].G0_iw
                ar['DMFT_results']['last_iter']['Gimp_tau_'+str(icrsh)] = S[icrsh].G_tau_man
                if solver_parameters["measure_G_l"]:
                    ar['DMFT_results']['last_iter']['Gimp_l_'+str(icrsh)] = S[icrsh].G_l_man
                ar['DMFT_results']['last_iter']['Gimp_iw_'+str(icrsh)] = S[icrsh].G_iw
                ar['DMFT_results']['last_iter']['Sigma_iw_'+str(icrsh)] = S[icrsh].Sigma_iw

            # save to h5 archive every h5_save_freq iterations
            if ( it % general_parameters['h5_save_freq'] == 0):
                ar['DMFT_results'].create_group('it_'+str(it))
                ar['DMFT_results']['it_'+str(it)]['chemical_potential'] = SK.chemical_potential
                ar['DMFT_results']['it_'+str(it)]['DC_pot'] = SK.dc_imp
                ar['DMFT_results']['it_'+str(it)]['DC_energ'] = SK.dc_energ
                ar['DMFT_results']['it_'+str(it)]['dens_mat_pre'] = density_mat_pre
                ar['DMFT_results']['it_'+str(it)]['dens_mat_post'] = density_mat
                for icrsh in range(SK.n_inequiv_shells):
                    ar['DMFT_results']['it_'+str(it)]['G0_iw'+str(icrsh)] = S[icrsh].G0_iw
                    ar['DMFT_results']['it_'+str(it)]['Gimp_tau_'+str(icrsh)] = S[icrsh].G_tau_man
                    if solver_parameters["measure_G_l"]:
                        ar['DMFT_results']['it_'+str(it)]['Gimp_l_'+str(icrsh)] = S[icrsh].G_l_man
                    ar['DMFT_results']['it_'+str(it)]['Gimp_iw_'+str(icrsh)] = S[icrsh].G_iw
                    ar['DMFT_results']['it_'+str(it)]['Sigma_iw_'+str(icrsh)] = S[icrsh].Sigma_iw

        mpi.barrier()


        # if we do a CSC calculation we need always an updated GAMMA file
        E_bandcorr = 0.0
        if general_parameters['csc']:
            # handling the density correction for fcsc calculations
            dN, d, E_bandcorr = SK.calc_density_correction(filename = 'GAMMA',dm_type='vasp')

        # for a one shot calculation we are using our own method
        if not general_parameters['csc'] and general_parameters['calc_energies'] == True:
            E_bandcorr = calc_bandcorr_man(general_parameters, SK, E_kin_dft)

        # calculate observables and write them to file
        if mpi.is_master_node():
            '\n *** calculation of observables ***'
            observables = calc_obs(observables,
                        general_parameters,
                        solver_parameters,
                        it,
                        S,
                        h_int,
                        dft_mu,
                        previous_mu,
                        SK,
                        G_loc_all_dft,
                        density_mat_dft,
                        density_mat,
                        shell_multiplicity,
                        E_bandcorr)

            write_obs(observables,SK,general_parameters)

            # write the new observable array to h5 archive
            ar['DMFT_results']['observables'] = observables

            print('*** iteration finished ***')

            # print out of the energies
            if general_parameters['calc_energies']:
                print('')
                print("="*60)
                print('summary of energetics:')
                print("total energy: ", observables['E_tot'][-1])
                print("DFT energy: ", observables['E_dft'][-1])
                print("correllation energy: ", observables['E_corr_en'][-1])
                print("DFT band correction: ", observables['E_bandcorr'][-1])
                print("="*60)
                print('')

            # print out summary of occupations per impurity
            print("="*60)
            print('summary of occupations: ')
            for icrsh in range(SK.n_inequiv_shells):
                print('total occupany of impurity '+str(icrsh)+':'+"{:7.4f}".format(observables['imp_occ'][icrsh]['up'][-1]+observables['imp_occ'][icrsh]['down'][-1]))
            for icrsh in range(SK.n_inequiv_shells):
                print('G(beta/2) occ of impurity '+str(icrsh)+':'+"{:8.4f}".format(observables['imp_gb2'][icrsh]['up'][-1]+observables['imp_gb2'][icrsh]['down'][-1]))
            print("="*60)
            print('')

            # if a magnetic calculation is done print out a summary of up/down occ
            if general_parameters['magnetic']:
                occ = {}
                occ['up'] = 0.0
                occ['down'] = 0.0
                print('')
                print("="*60)
                print('\n *** summary of magnetic occupations: ***')
                for icrsh in range(SK.n_inequiv_shells):
                        for spin in ['up','down']:
                            temp = observables['imp_occ'][icrsh][spin][-1]
                            print('imp '+str(icrsh)+' spin '+spin+': '+"{:6.4f}".format(temp))
                            occ[spin] += temp

                print('total spin up   occ: '+"{:6.4f}".format(occ['up']))
                print('total spin down occ: '+"{:6.4f}".format(occ['down']))
                print("="*60)
                print('')

        # check for convergency and stop if criteria is reached
        if it == 1 or it == iteration_offset+1:
            converged = False
        std_dev = 0.0
        if it == 1 and general_parameters['occ_conv_crit'] > 0.0 and mpi.is_master_node():
            conv_file.write('std_dev occ for each impurity \n')
        if it >= general_parameters['occ_conv_it'] and general_parameters['occ_conv_crit'] > 0.0:
            if mpi.is_master_node():
                converged, std_dev = toolset.check_convergence(SK,general_parameters,observables)
                conv_file.write("{:3d}".format(it))
                for icrsh in range(SK.n_inequiv_shells):
                    conv_file.write("{:10.6f}".format(std_dev[icrsh]))
                conv_file.write('\n')
                conv_file.flush()
            converged = mpi.bcast(converged)
            std_dev = mpi.bcast(std_dev)

        # check for convergency and if wanted do the sampling dmft iterations.
        if converged == True and sampling == False:
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
            mpi.report("#"*80)
            dmft_looping = False

        # iteration counter
        it += 1

    if converged and general_parameters['csc']:
        # is there a smoother way to stop both vasp and triqs from running after convergency is reached?
        if mpi.is_master_node():
            f_stop = open('STOPCAR', 'wt')
            f_stop.write("LABORT = .TRUE.\n")
            f_stop.close()
            del ar
        mpi.MPI.COMM_WORLD.Abort(1)

    if mpi.is_master_node():
        if general_parameters['occ_conv_crit'] > 0.0:
            conv_file.close()
    mpi.barrier()

    # close the h5 archive
    if mpi.is_master_node():
        if 'h5_archive' in locals():
            del h5_archive

    return observables
