# contains all functions related to the observables
# prep_observables
# prepare_obs_files
# calc_obs
# write_obs
# calc_dft_kin_en
# calc_bandcorr_man

# system
import numpy as np

# triqs
import pytriqs.utility.mpi as mpi
try:
    # TRIQS 2.0
    from pytriqs.gf import GfImTime
    from pytriqs.atom_diag import trace_rho_op
    from pytriqs.gf.descriptors import InverseFourier
except ImportError:
    # TRIQS 1.4
    from pytriqs.applications.impurity_solvers.cthyb import *
    from pytriqs.applications.dft.sumk_dft import *
    from pytriqs.applications.dft.sumk_dft_tools import *
    from pytriqs.applications.dft.converters.vasp_converter import *
    from pytriqs.applications.dft.converters.plovasp.vaspio import VaspData
    import pytriqs.applications.dft.converters.plovasp.converter as plo_converter
    from pytriqs.gf.local import *
    

def prep_observables(general_parameters, h5_archive):
    """
    prepares the observable arrays and files for the DMFT calculation

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    h5_archive: hdf archive instance
        hdf archive for calculation

    __Returns:__
    observables : dict
        observable array for calculation
    """

    # determine number of impurities
    n_inequiv_shells = h5_archive['dft_input']['n_inequiv_shells']

    # check for previous iterations
    obs_prev = []
    if 'observables' in h5_archive['DMFT_results']:
        obs_prev = h5_archive['DMFT_results']['observables']

    # prepare observable dicts
    if len(obs_prev) > 0:
        observables = obs_prev
    else:
        observables = dict()
        observables['iteration'] = []
        observables['mu'] = []
        observables['E_tot'] = []
        observables['E_bandcorr'] = []
        observables['E_int'] = []
        observables['E_int_rho'] = []
        observables['E_corr_en'] = []
        observables['E_dft'] = []
        observables['E_DC'] = []
        observables['orb_gb2'] = []
        observables['imp_gb2'] = []
        observables['orb_occ'] = []
        observables['imp_occ'] = []
        observables['rho'] = []
        observables['h_loc'] = []
        observables['h_loc_diag'] = []
        for icrsh in range(0,n_inequiv_shells):

            observables['E_int'].append([])
            observables['E_int_rho'].append([])
            observables['E_DC'].append([])
            observables['rho'].append([])
            observables['h_loc'].append([])
            observables['h_loc_diag'].append([])
            observables['orb_gb2'].append({'up': [], 'down': []})
            observables['imp_gb2'].append({'up': [], 'down': []})
            observables['orb_occ'].append({'up': [], 'down': []})
            observables['imp_occ'].append({'up': [], 'down': []})

    # prepare observable files
    if not 'iteration_count' in h5_archive['DMFT_results']:
        prepare_obs_files(general_parameters,n_inequiv_shells)

    return observables

def prepare_obs_files(general_parameters,n_inequiv_shells):
    """
    prepares the observable files for writing

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    n_inequiv_shells : int
        number of impurities for calculations


    __Returns:__
    nothing

    """

    if general_parameters['magnetic']:
        observables_up = []
        observables_down = []
    else:
        observables = []

    for icrsh in range(n_inequiv_shells):
        if general_parameters['magnetic']:
            print('observables for impurity '+str(icrsh)+' are written to file observables_imp'+str(icrsh)+'_up.dat and _down.dat')
        else:
            print('observables for impurity '+str(icrsh)+' are written to file observables_imp'+str(icrsh)+'.dat')
        # only print header if no previous iterations are found

        # if magnetic calculation is done create two obs files per imp
        if general_parameters['magnetic']:
            observables_up.append(open(general_parameters['jobname']+'/'+'observables_imp'+str(icrsh)+'_up.dat', 'w'))
            observables_down.append(open(general_parameters['jobname']+'/'+'observables_imp'+str(icrsh)+'_down.dat', 'w'))

            observables_up[icrsh].write(' it    |  mu   |  G(beta/2) per orbital  |  orbital occs up  |  impurity occ up channel')
            observables_down[icrsh].write(' it    |  mu   |  G(beta/2) per orbital  |  orbital occs down  |  impurity occ down channel')
            if general_parameters['calc_energies']:
                observables_up[icrsh].write(' | E_tot E_DFT E_bandcorr E_int_imp E_DC \n')
                observables_down[icrsh].write(' | E_tot E_DFT E_bandcorr E_int_imp E_DC \n')
            else:
                observables_up[icrsh].write('\n')
                observables_down[icrsh].write('\n')

            observables_up[icrsh].close()
            observables_down[icrsh].close()

        else:
            observables.append(open(general_parameters['jobname']+'/'+'observables_imp'+str(icrsh)+'.dat', 'w'))

            observables[icrsh].write(' it    |  mu   |  G(beta/2) per orbital  |  orbital occs up+down  |  impurity occ')
            if general_parameters['calc_energies']:
                observables[icrsh].write(' | E_tot E_DFT E_bandcorr E_int_imp E_DC \n')
            else:
                observables[icrsh].write('\n')

            observables[icrsh].close()

    return

def calc_obs(observables, general_parameters, solver_parameters, it, S, h_int, dft_mu, previous_mu, SK, G_loc_all_dft, density_mat_dft, density_mat, shell_multiplicity, E_dft, E_bandcorr, E_int, E_corr_en):
    """
    calculates the observables for given Input, I decided to calculate the observables
    not adhoc since it should be done only once by the master_node

    Parameters
    ----------
    observables : observable arrays/dicts

    general_parameters : general parameters as a dict

    solver_parameters : solver parameters as a dict

    it : iteration counter

    S : Solver instances

    h_int : interaction hamiltonian

    dft_mu : dft chemical potential

    previous_mu : dmft chemical potential for which the calculation was just done

    SK : SumK Object instances

    G_loc_all_dft : Gloc from DFT for G(beta/2)

    density_mat_dft : occupations from DFT

    density_mat : DMFT occupations

    shell_multiplicity : degeneracy of impurities

    E_dft: latest dft energy

    E_bandcorr : E_kin_dmft - E_kin_dft, either calculated man or from SK method if CSC

    E_int : array of interaction energies for each imp

    E_corr_en : total interaction energy for whole unit cell - Impurity DC energy

    __Returns:__

    observables: list of dicts
    """

    # write the DFT obs a 0 iteration
    if it == 1:
        observables['iteration'].append(0)
        observables['mu'].append(float(dft_mu))
        observables['E_bandcorr'].append(0.0)
        observables['E_corr_en'].append(0.0)
        observables['E_dft'].append(E_dft)
        observables['E_tot'].append(E_dft)

        for icrsh in range(0,SK.n_inequiv_shells):
            # determine number of orbitals per shell
            observables['E_int'][icrsh].append(0.0)
            observables['E_int_rho'][icrsh].append(0.0)
            if (general_parameters['dc_type'] >= 0):
                observables['E_DC'][icrsh].append(shell_multiplicity[icrsh]*SK.dc_energ[SK.inequiv_to_corr[icrsh]])
            else:
                observables['E_DC'][icrsh].append(0.0)

            gb2_imp_up = 0.0
            gb2_imp_down = 0.0
            gb2_orb_up = []
            gb2_orb_down = []
            occ_imp_up = 0.0
            occ_imp_down = 0.0
            occ_orb_up = []
            occ_orb_down = []

            # iterate over all spin channels and add the to up or down
            for spin_channel, elem in SK.gf_struct_solver[icrsh].iteritems():

                # G(beta/2)
                G_tau = GfImTime(indices=S[icrsh].G_tau_man[spin_channel].indices,
                                beta=general_parameters['beta'])
                G_tau << InverseFourier(G_loc_all_dft[icrsh][spin_channel])

                mesh_len = len(G_tau.data)
                mesh_mid = int(len(G_tau.data)/2)
                # since G(tau) has always 10001 values we are sampling +-10 values
                # hard coded, for beta=40 this corresponds to approx +-0.05
                samp = 10
                # we are sampling a few values around G(beta/2)
                gg = G_tau.data[mesh_mid-samp:mesh_mid+samp]
                # taking the diagonal elements of the sum of the G(beta/2) matrices
                gb2_list = np.diagonal(np.real(sum(gg)/float(2*samp)))

                for orb, value in enumerate(gb2_list):
                    if 'up' in spin_channel:
                        gb2_orb_up.append(value)
                        gb2_imp_up += value
                    else:
                        gb2_orb_down.append(value)
                        gb2_imp_down += value

                # occupation per orbital
                den_mat = density_mat_dft[icrsh][spin_channel]
                for i in range(0,len(np.real(den_mat)[0,:])):
                    if 'up' in spin_channel:
                        occ_orb_up.append(np.real(den_mat)[i,i])
                        occ_imp_up += np.real(den_mat)[i,i]
                    else:
                        occ_orb_down.append(np.real(den_mat)[i,i])
                        occ_imp_down += np.real(den_mat)[i,i]

            # adding those values to the observable object
            observables['orb_gb2'][icrsh]['up'].append(gb2_orb_up)
            observables['orb_gb2'][icrsh]['down'].append(gb2_orb_down)

            observables['imp_gb2'][icrsh]['up'].append(gb2_imp_up)
            observables['imp_gb2'][icrsh]['down'].append(gb2_imp_down)

            observables['orb_occ'][icrsh]['up'].append(occ_orb_up)
            observables['orb_occ'][icrsh]['down'].append(occ_orb_down)

            observables['imp_occ'][icrsh]['up'].append(occ_imp_up)
            observables['imp_occ'][icrsh]['down'].append(occ_imp_down)

        # write the DFT observables to the files
        write_obs(observables,SK,general_parameters)


    # now the normal output from each iteration
    observables['iteration'].append(it)
    observables['mu'].append(float(previous_mu))
    observables['E_bandcorr'].append(E_bandcorr)
    observables['E_corr_en'].append(E_corr_en)
    observables['E_dft'].append(E_dft)

    # calc total energy
    E_tot = E_dft + E_bandcorr + E_corr_en
    observables['E_tot'].append(E_tot)

    for icrsh in range(0,SK.n_inequiv_shells):
        observables['E_int'][icrsh].append(float(shell_multiplicity[icrsh]*E_int[icrsh]))
        if (general_parameters['dc_type'] >= 0):
            observables['E_DC'][icrsh].append(shell_multiplicity[icrsh]*SK.dc_energ[SK.inequiv_to_corr[icrsh]])
        else:
            observables['E_DC'][icrsh].append(0.0)

        gb2_imp_up = 0.0
        gb2_imp_down = 0.0
        gb2_orb_up = []
        gb2_orb_down = []
        occ_imp_up = 0.0
        occ_imp_down = 0.0
        occ_orb_up = []
        occ_orb_down = []

        # if density matrix was measured store result in observables
        if solver_parameters["measure_density_matrix"] and mpi.is_master_node():
            mpi.report("\nextracting the impurity density matrix")
            # Extract accumulated density matrix
            observables["rho"][icrsh].append( S[icrsh].density_matrix )

            # storing the local Hamiltonian
            observables["h_loc"][icrsh].append( S[icrsh].h_loc )

            # Object containing eigensystem of the local Hamiltonian
            observables["h_loc_diag"][icrsh].append( S[icrsh].h_loc_diagonalization )

            E_h_int = trace_rho_op(observables["rho"][icrsh][-1],
                                    h_int[icrsh],
                                    observables["h_loc_diag"][icrsh][-1])

            observables['E_int_rho'][icrsh].append(float(shell_multiplicity[icrsh]*E_h_int))

        # iterate over all spin channels and add the to up or down
        for spin_channel, elem in SK.gf_struct_solver[icrsh].iteritems():

            # G(beta/2)
            mesh_len = len(S[icrsh].G_tau_man[spin_channel].data)
            mesh_mid = int(len(S[icrsh].G_tau_man[spin_channel].data)/2)
            # since G(tau) has always 10001 values we are sampling +-10 values
            # hard coded, for beta=40 this corresponds to approx +-0.05
            samp = 10
            # we are sampling a few values around G(beta/2+-0.05)
            gg = S[icrsh].G_tau_man[spin_channel].data[mesh_mid-samp:mesh_mid+samp]
            # taking the diagonal elements of the sum of the G(beta/2) matrices
            gb2_list = np.diagonal(np.real(sum(gg)/float(2*samp)))

            for orb, value in enumerate(gb2_list):
                if 'up' in spin_channel:
                    gb2_orb_up.append(value)
                    gb2_imp_up += value
                else:
                    gb2_orb_down.append(value)
                    gb2_imp_down += value

            # occupation per orbital and impurity
            den_mat = density_mat[icrsh][spin_channel]
            for i in range(0,len(np.real(den_mat)[0,:])):
                if 'up' in spin_channel:
                    occ_orb_up.append(np.real(den_mat)[i,i])
                    occ_imp_up += np.real(den_mat)[i,i]
                else:
                    occ_orb_down.append(np.real(den_mat)[i,i])
                    occ_imp_down += np.real(den_mat)[i,i]

        # adding those values to the observable object
        observables['orb_gb2'][icrsh]['up'].append(gb2_orb_up)
        observables['orb_gb2'][icrsh]['down'].append(gb2_orb_down)

        observables['imp_gb2'][icrsh]['up'].append(gb2_imp_up)
        observables['imp_gb2'][icrsh]['down'].append(gb2_imp_down)

        observables['orb_occ'][icrsh]['up'].append(occ_orb_up)
        observables['orb_occ'][icrsh]['down'].append(occ_orb_down)

        observables['imp_occ'][icrsh]['up'].append(occ_imp_up)
        observables['imp_occ'][icrsh]['down'].append(occ_imp_down)

    return observables

def write_obs(observables,SK,general_parameters):
    """
    writes the last entries of the observable arrays to the files

    Parameters
    ----------
    observables : list of dicts
        observable arrays/dicts

    SK : SumK Object instances

    general_parameters : dict

    __Returns:__

    nothing

    """

    if general_parameters['magnetic']:
        obs_files = {}
        for spin in ['up','down']:
            obs_files[spin] = []

    else:
        obs_files = []

    # open obs files
    nfiles = SK.n_inequiv_shells
    for ifile in range(0,nfiles):
        if general_parameters['magnetic']:
            obs_files['up'].append(open(general_parameters['jobname']+'/'+
                                    'observables_imp'+str(ifile)+'_up.dat', 'a'))
            obs_files['down'].append(open(general_parameters['jobname']+'/'+
                                    'observables_imp'+str(ifile)+'_down.dat', 'a'))
        else:
            obs_files.append(open(general_parameters['jobname']+'/'+
                            'observables_imp'+str(ifile)+'.dat', 'a'))

    for icrsh in range(0,SK.n_inequiv_shells):

        if general_parameters['magnetic']:
            for spin in ['up','down']:
                obs_files[spin][icrsh].write("{:3d}".format(observables['iteration'][-1]))
                obs_files[spin][icrsh].write('  ')
                obs_files[spin][icrsh].write("{:10.5f}".format(observables['mu'][-1]))
                obs_files[spin][icrsh].write('  ')

                for i, item in enumerate(observables['orb_gb2'][icrsh][spin][-1]):
                    obs_files[spin][icrsh].write("{:10.5f}".format(item))
                    obs_files[spin][icrsh].write('  ')

                for i, item in enumerate(observables['orb_occ'][icrsh][spin][-1]):
                    obs_files[spin][icrsh].write("{:10.5f}".format(item))
                    obs_files[spin][icrsh].write('  ')

                obs_files[spin][icrsh].write("{:10.5f}".format(observables['imp_occ'][icrsh][spin][-1]))

                if general_parameters['calc_energies']:
                    obs_files[spin][icrsh].write('  ')
                    obs_files[spin][icrsh].write("{:10.5f}".format(observables['E_tot'][-1]))
                    obs_files[spin][icrsh].write('  ')
                    obs_files[spin][icrsh].write("{:10.5f}".format(observables['E_dft'][-1]))
                    obs_files[spin][icrsh].write('  ')
                    obs_files[spin][icrsh].write("{:10.5f}".format(observables['E_bandcorr'][-1]))
                    obs_files[spin][icrsh].write('  ')
                    obs_files[spin][icrsh].write("{:10.5f}".format(observables['E_int'][icrsh][-1]))
                    obs_files[spin][icrsh].write('  ')
                    obs_files[spin][icrsh].write("{:10.5f}".format(observables['E_DC'][icrsh][-1]))

                obs_files[spin][icrsh].write('\n')
        else:
            # adding up the spin channels
            obs_files[icrsh].write("{:3d}".format(observables['iteration'][-1]))
            obs_files[icrsh].write('  ')
            obs_files[icrsh].write("{:10.5f}".format(observables['mu'][-1]))
            obs_files[icrsh].write('  ')

            for i, item in enumerate(observables['orb_gb2'][icrsh]['up'][-1]):
                val = (observables['orb_gb2'][icrsh]['up'][-1][i]+
                       observables['orb_gb2'][icrsh]['down'][-1][i])
                obs_files[icrsh].write("{:10.5f}".format(val))
                obs_files[icrsh].write('  ')

            for i, item in enumerate(observables['orb_occ'][icrsh]['up'][-1]):
                val = (observables['orb_occ'][icrsh]['up'][-1][i]+
                       observables['orb_occ'][icrsh]['down'][-1][i])
                obs_files[icrsh].write("{:10.5f}".format(val))
                obs_files[icrsh].write('  ')

            val = observables['imp_occ'][icrsh]['up'][-1]+observables['imp_occ'][icrsh]['down'][-1]
            obs_files[icrsh].write("{:10.5f}".format(val))

            if general_parameters['calc_energies']:
                obs_files[icrsh].write('  ')
                obs_files[icrsh].write("{:10.5f}".format(observables['E_tot'][-1]))
                obs_files[icrsh].write('  ')
                obs_files[icrsh].write("{:10.5f}".format(observables['E_dft'][-1]))
                obs_files[icrsh].write('  ')
                obs_files[icrsh].write("{:10.5f}".format(observables['E_bandcorr'][-1]))
                obs_files[icrsh].write('  ')
                obs_files[icrsh].write("{:10.5f}".format(observables['E_int'][icrsh][-1]))
                obs_files[icrsh].write('  ')
                obs_files[icrsh].write("{:10.5f}".format(observables['E_DC'][icrsh][-1]))

            obs_files[icrsh].write('\n')

    # closing the files
    for ifile in range(0,nfiles):
        if general_parameters['magnetic']:
            for spin in ['up','down']:
                obs_files[spin][ifile].close()
        else:
            obs_files[ifile].close()

    return

def calc_dft_kin_en(general_parameters, SK, dft_mu):
    """
    Calculates the kinetic energy from DFT for target states

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict

    SK : SumK Object instances

    dft_mu: float
        DFT fermi energy


    __Returns:__
    E_kin_dft: float
        kinetic energy from DFT

    """

    H_ks = SK.hopping
    num_kpts = SK.n_k
    E_kin = 0.0+0.0j
    ikarray = np.array(range(SK.n_k))
    for ik in mpi.slice_array(ikarray):
        nb = int(SK.n_orbitals[ik])
        # calculate lattice greens function
        G_iw_lat = SK.lattice_gf(ik,
                                beta=general_parameters['beta'],
                                with_Sigma=False,mu=dft_mu).copy()
        # calculate G(beta) via the function density, which is the same as fourier trafo G(w) and taking G(b)
        G_iw_lat_beta = G_iw_lat.density()
        # Doing the formula
        E_kin += np.trace(np.dot(H_ks[ik,0,:nb,:nb],G_iw_lat_beta['up'][:,:])) + np.trace(np.dot(H_ks[ik,0,:nb,:nb],G_iw_lat_beta['down'][:,:]))
    E_kin = float(E_kin.real)

    # collect data and put into E_kin_dft
    E_kin_dft = mpi.all_reduce(mpi.world, E_kin, lambda x,y : x+y)
    mpi.barrier()
    # E_kin should be divided by the number of k-points
    E_kin_dft = E_kin_dft/num_kpts

    if mpi.is_master_node():
        print('Kinetic energy contribution dft part: '+str(E_kin_dft))

    return E_kin_dft

def calc_bandcorr_man(general_parameters, SK, E_kin_dft):
    """
    Calculates the correlated kinetic energy from DMFT for target states
    and then determines the band correction energy

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict

    SK : SumK Object instances

    E_kin_dft: float
        kinetic energy from DFT


    __Returns:__
    E_bandcorr: float
        band energy correction E_kin_dmft - E_kin_dft

    """
    E_kin_dmft = 0.0+0.0j
    E_kin = 0.0+0.0j
    H_ks = SK.hopping
    num_kpts = SK.n_k

    # kinetic energy from dmft lattice Greens functions
    ikarray = np.array(range(SK.n_k))
    for ik in mpi.slice_array(ikarray):
        nb = int(SK.n_orbitals[ik])
        # calculate lattice greens function
        G_iw_lat = SK.lattice_gf(ik,beta=general_parameters['beta'],with_Sigma=True,with_dc=True).copy()
        # calculate G(beta) via the function density, which is the same as fourier trafo G(w) and taking G(b)
        G_iw_lat_beta = G_iw_lat.density()
        # Doing the formula
        E_kin += np.trace(np.dot(H_ks[ik,0,:nb,:nb],G_iw_lat_beta['up'][:,:])) + np.trace(np.dot(H_ks[ik,0,:nb,:nb],G_iw_lat_beta['down'][:,:]))
    E_kin = float(E_kin.real)

    # collect data and put into E_kin_dmft
    E_kin_dmft = mpi.all_reduce(mpi.world, E_kin, lambda x,y : x+y)
    mpi.barrier()
    # E_kin should be divided by the number of k-points
    E_kin_dmft = E_kin_dmft/num_kpts

    if mpi.is_master_node():
        print('Kinetic energy contribution dmft part: '+str(E_kin_dmft))

    E_bandcorr = E_kin_dmft - E_kin_dft

    return E_bandcorr
