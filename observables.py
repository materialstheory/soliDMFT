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
from pytriqs.gf import GfImTime
from pytriqs.atom_diag import trace_rho_op
from pytriqs.gf.descriptors import Fourier


import toolset

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
        observables['E_int'] = [[] for _ in range(n_inequiv_shells)]
        observables['E_corr_en'] = []
        observables['E_dft'] = []
        observables['E_DC'] = [[] for _ in range(n_inequiv_shells)]
        observables['orb_gb2'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
        observables['imp_gb2'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
        observables['orb_occ'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
        observables['imp_occ'] = [{'up': [], 'down': []} for _ in range(n_inequiv_shells)]
        observables['rho'] = [[] for _ in range(n_inequiv_shells)]
        observables['h_loc'] = [[] for _ in range(n_inequiv_shells)]
        observables['h_loc_diag'] = [[] for _ in range(n_inequiv_shells)]

    return observables

def prepare_obs_files(general_parameters, n_inequiv_shells, n_orbitals):
    """
    Prepares the observable files for writing

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    n_inequiv_shells : int
        number of impurities for calculations


    __Returns:__
    nothing

    """
    header_energy_mask = ' | {:>10} | {:>10}   {:>10}   {:>10}   {:>10}'
    header_energy = header_energy_mask.format('E_tot', 'E_DFT', 'E_bandcorr', 'E_int_imp', 'E_DC')

    for icrsh in range(n_inequiv_shells):
        number_spaces = max(10*n_orbitals[icrsh] + 3*(n_orbitals[icrsh]-1), 21)
        header_basic_mask = '{{:>3}} | {{:>10}} | {{:>{}}} | {{:>{}}} | {{:>17}}'.format(number_spaces, number_spaces)

        # If magnetic calculation is done create two obs files per imp
        if general_parameters['magnetic']:
            for spin in ('up', 'down'):
                header = header_basic_mask.format('it', 'mu', 'G(beta/2) per orbital', 'orbital occs '+spin, 'impurity occ '+spin)

                if general_parameters['calc_energies']:
                    header += header_energy

                file_name = '{}/observables_imp{}_{}.dat'.format(general_parameters['jobname'], icrsh, spin)
                with open(file_name, 'w') as obs_file:
                    obs_file.write(header + '\n')
        else:
            header = header_basic_mask.format('it', 'mu', 'G(beta/2) per orbital', 'orbital occs up+down', 'impurity occ')

            if general_parameters['calc_energies']:
                header += header_energy

            file_name = '{}/observables_imp{}.dat'.format(general_parameters['jobname'], icrsh)
            with open(file_name, 'w') as obs_file:
                obs_file.write(header + '\n')

def calc_obs(observables, general_parameters, solver_parameters, it, solvers, h_int, dft_mu, previous_mu, sum_k, G_loc_all_dft, density_mat_dft, density_mat, shell_multiplicity, E_bandcorr):
    """
    calculates the observables for given Input, I decided to calculate the observables
    not adhoc since it should be done only once by the master_node

    Parameters
    ----------
    observables : observable arrays/dicts

    general_parameters : general parameters as a dict

    solver_parameters : solver parameters as a dict

    it : iteration counter

    solvers : Solver instances

    h_int : interaction hamiltonian

    dft_mu : dft chemical potential

    previous_mu : dmft chemical potential for which the calculation was just done

    sum_k : SumK Object instances

    G_loc_all_dft : Gloc from DFT for G(beta/2)

    density_mat_dft : occupations from DFT

    density_mat : DMFT occupations

    shell_multiplicity : degeneracy of impurities

    E_bandcorr : E_kin_dmft - E_kin_dft, either calculated man or from sum_k method if CSC

    __Returns:__

    observables: list of dicts
    """

    # init energy values
    E_dft = 0.0
    E_int = np.zeros(sum_k.n_inequiv_shells)
    E_corr_en = 0.0
    E_DC = 0.0

    if general_parameters['csc']:
        # Read energy from OSZICAR
        E_dft = toolset.get_dft_energy()

    # write the DFT obs a 0 iteration
    if it == 1:
        observables['iteration'].append(0)
        observables['mu'].append(float(dft_mu))
        observables['E_bandcorr'].append(0.0)
        observables['E_corr_en'].append(0.0)
        observables['E_dft'].append(E_dft)

        for icrsh in range(sum_k.n_inequiv_shells):
            # determine number of orbitals per shell
            observables['E_int'][icrsh].append(0.0)
            if general_parameters['dc_type'] >= 0:
                observables['E_DC'][icrsh].append(shell_multiplicity[icrsh]*sum_k.dc_energ[sum_k.inequiv_to_corr[icrsh]])
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

            for icrsh in range(sum_k.n_inequiv_shells):
                E_DC += shell_multiplicity[icrsh]*sum_k.dc_energ[sum_k.inequiv_to_corr[icrsh]]

            # for it 0 we just subtract E_DC from E_DFT
            observables['E_tot'].append(E_dft - E_DC)

            # iterate over all spin channels and add the to up or down
            for spin_channel in sum_k.gf_struct_solver[icrsh]:

                # G(beta/2)
                G_tau = GfImTime(indices=solvers[icrsh].G_tau_man[spin_channel].indices,
                                 beta=general_parameters['beta'])
                G_tau << Fourier(G_loc_all_dft[icrsh][spin_channel])

                mesh_mid = int(len(G_tau.data)/2)
                # since G(tau) has always 10001 values we are sampling +-10 values
                # hard coded, for beta=40 this corresponds to approx +-0.05
                samp = 10
                # we are sampling a few values around G(beta/2)
                gg = G_tau.data[mesh_mid-samp:mesh_mid+samp]
                # taking the diagonal elements of the sum of the G(beta/2) matrices
                gb2_list = np.diagonal(np.real(sum(gg)/float(2*samp)))

                for value in gb2_list:
                    if 'up' in spin_channel:
                        gb2_orb_up.append(value)
                        gb2_imp_up += value
                    else:
                        gb2_orb_down.append(value)
                        gb2_imp_down += value

                # occupation per orbital
                den_mat = density_mat_dft[icrsh][spin_channel]
                for i in range(len(np.real(den_mat)[0, :])):
                    if 'up' in spin_channel:
                        occ_orb_up.append(np.real(den_mat)[i, i])
                        occ_imp_up += np.real(den_mat)[i, i]
                    else:
                        occ_orb_down.append(np.real(den_mat)[i, i])
                        occ_imp_down += np.real(den_mat)[i, i]

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
        write_obs(observables, sum_k, general_parameters)


    # now the normal output from each iteration
    observables['iteration'].append(it)
    observables['mu'].append(float(previous_mu))
    observables['E_bandcorr'].append(E_bandcorr)
    observables['E_dft'].append(E_dft)

    # if density matrix was measured store result in observables
    if solver_parameters["measure_density_matrix"]:
        for icrsh in range(sum_k.n_inequiv_shells):
            if icrsh == 0:
                mpi.report("\nextracting the impurity density matrix")
            # Extract accumulated density matrix
            observables["rho"][icrsh].append( solvers[icrsh].density_matrix )

            # storing the local Hamiltonian
            observables["h_loc"][icrsh].append( solvers[icrsh].h_loc )

            # Object containing eigensystem of the local Hamiltonian
            observables["h_loc_diag"][icrsh].append( solvers[icrsh].h_loc_diagonalization )

    if general_parameters['calc_energies']:
        # dmft interaction energy with E_int = 0.5 * Tr[Sigma * G]
        for icrsh in range(sum_k.n_inequiv_shells):
            if solver_parameters["measure_density_matrix"]:
                E_int[icrsh] = trace_rho_op(observables["rho"][icrsh][-1],
                                        h_int[icrsh],
                                        observables["h_loc_diag"][icrsh][-1])
            else:
                #calc energy for given S and G
                warning = ( "!-------------------------------------------------------------------------------------------!\n"
                            "! WARNING: calculating interaction energy using Migdal formula                              !\n"
                            "! consider turning on measure density matrix to use the more stable trace_rho_op function   !\n"
                            "!-------------------------------------------------------------------------------------------!" )
                print(warning)

                E_int[icrsh] = 0.5 * (solvers[icrsh].G_iw* solvers[icrsh].Sigma_iw).total_density()

            observables['E_int'][icrsh].append(float(shell_multiplicity[icrsh]*E_int[icrsh]))
            E_corr_en += shell_multiplicity[icrsh]*E_int[icrsh] - shell_multiplicity[icrsh]*sum_k.dc_energ[sum_k.inequiv_to_corr[icrsh]]


    observables['E_corr_en'].append(E_corr_en)

    # calc total energy
    E_tot = E_dft + E_bandcorr + E_corr_en
    observables['E_tot'].append(E_tot)

    for icrsh in range(sum_k.n_inequiv_shells):
        if general_parameters['dc_type'] >= 0:
            observables['E_DC'][icrsh].append(shell_multiplicity[icrsh]*sum_k.dc_energ[sum_k.inequiv_to_corr[icrsh]])
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
        for spin_channel, _ in sum_k.gf_struct_solver[icrsh].iteritems():

            # G(beta/2)
#            mesh_len = len(solvers[icrsh].G_tau_man[spin_channel].data)
            mesh_mid = int(len(solvers[icrsh].G_tau_man[spin_channel].data)/2)
            # since G(tau) has always 10001 values we are sampling +-10 values
            # hard coded, for beta=40 this corresponds to approx +-0.05
            samp = 10
            # we are sampling a few values around G(beta/2+-0.05)
            gg = solvers[icrsh].G_tau_man[spin_channel].data[mesh_mid-samp:mesh_mid+samp]
            # taking the diagonal elements of the sum of the G(beta/2) matrices
            gb2_list = np.diagonal(np.real(sum(gg)/float(2*samp)))

            for value in gb2_list:
                if 'up' in spin_channel:
                    gb2_orb_up.append(value)
                    gb2_imp_up += value
                else:
                    gb2_orb_down.append(value)
                    gb2_imp_down += value

            # occupation per orbital and impurity
            den_mat = density_mat[icrsh][spin_channel]
            for i in range(len(np.real(den_mat)[0, :])):
                if 'up' in spin_channel:
                    occ_orb_up.append(np.real(den_mat)[i, i])
                    occ_imp_up += np.real(den_mat)[i, i]
                else:
                    occ_orb_down.append(np.real(den_mat)[i, i])
                    occ_imp_down += np.real(den_mat)[i, i]

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

def write_obs(observables, sum_k, general_parameters):
    """
    writes the last entries of the observable arrays to the files

    Parameters
    ----------
    observables : list of dicts
        observable arrays/dicts

    sum_k : SumK Object instances

    general_parameters : dict

    __Returns:__

    nothing

    """

    n_inequiv_shells = sum_k.n_inequiv_shells
    n_orbitals = [len(observables['orb_gb2'][icrsh]['up'][0]) for icrsh in range(n_inequiv_shells)]

    # Prints header in first iteration, using len() to not remove DFT obs writing
    if len(observables['iteration']) == 1:
        prepare_obs_files(general_parameters, n_inequiv_shells, n_orbitals)

    for icrsh in range(n_inequiv_shells):
        if general_parameters['magnetic']:
            for spin in ('up', 'down'):
                line = '{:3d} | '.format(observables['iteration'][-1])
                line += '{:10.5f} | '.format(observables['mu'][-1])

                if n_orbitals[icrsh] == 1:
                    line += ' '*11
                for item in observables['orb_gb2'][icrsh][spin][-1]:
                    line += '{:10.5f}   '.format(item)
                line = line[:-3] + ' | '

                if n_orbitals[icrsh] == 1:
                    line += ' '*11
                for item in observables['orb_occ'][icrsh][spin][-1]:
                    line += '{:10.5f}   '.format(item)
                line = line[:-3] + ' | '

                line += '{:17.5f}'.format(observables['imp_occ'][icrsh][spin][-1])

                if general_parameters['calc_energies']:
                    line += ' | {:10.5f}'.format(observables['E_tot'][-1])
                    line += ' | {:10.5f}'.format(observables['E_dft'][-1])
                    line += '   {:10.5f}'.format(observables['E_bandcorr'][-1])
                    line += '   {:10.5f}'.format(observables['E_int'][icrsh][-1])
                    line += '   {:10.5f}'.format(observables['E_DC'][icrsh][-1])

                file_name = '{}/observables_imp{}_{}.dat'.format(general_parameters['jobname'], icrsh, spin)
                with open(file_name, 'a') as obs_file:
                    obs_file.write(line + '\n')
        else:
            line = '{:3d} | '.format(observables['iteration'][-1])
            line += '{:10.5f} | '.format(observables['mu'][-1])

            # Adds spaces for header to fit in properly
            if n_orbitals[icrsh] == 1:
                line += ' '*11
            # Adds up the spin channels
            for val_up, val_down in zip(observables['orb_gb2'][icrsh]['up'][-1],
                                        observables['orb_gb2'][icrsh]['down'][-1]):
                line += '{:10.5f}   '.format(val_up + val_down)
            line = line[:-3] + ' | '

            # Adds spaces for header to fit in properly
            if n_orbitals[icrsh] == 1:
                line += ' '*11
            # Adds up the spin channels
            for val_up, val_down in zip(observables['orb_occ'][icrsh]['up'][-1],
                                        observables['orb_occ'][icrsh]['down'][-1]):
                line += '{:10.5f}   '.format(val_up + val_down)
            line = line[:-3] + ' | '

            # Adds up the spin channels
            val = observables['imp_occ'][icrsh]['up'][-1]+observables['imp_occ'][icrsh]['down'][-1]
            line += '{:17.5f}'.format(val)

            if general_parameters['calc_energies']:
                line += ' | {:10.5f}'.format(observables['E_tot'][-1])
                line += ' | {:10.5f}'.format(observables['E_dft'][-1])
                line += '   {:10.5f}'.format(observables['E_bandcorr'][-1])
                line += '   {:10.5f}'.format(observables['E_int'][icrsh][-1])
                line += '   {:10.5f}'.format(observables['E_DC'][icrsh][-1])

            file_name = '{}/observables_imp{}.dat'.format(general_parameters['jobname'], icrsh)
            with open(file_name, 'a') as obs_file:
                obs_file.write(line + '\n')

def calc_dft_kin_en(general_parameters, sum_k, dft_mu):
    """
    Calculates the kinetic energy from DFT for target states

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict

    sum_k : SumK Object instances

    dft_mu: float
        DFT fermi energy


    __Returns:__
    E_kin_dft: float
        kinetic energy from DFT

    """

    H_ks = sum_k.hopping
    num_kpts = sum_k.n_k
    E_kin = 0.0+0.0j
    ikarray = np.array(range(sum_k.n_k))
    for ik in mpi.slice_array(ikarray):
        nb = int(sum_k.n_orbitals[ik])
        # calculate lattice greens function
        G_iw_lat = sum_k.lattice_gf(ik,
                                    beta=general_parameters['beta'],
                                    with_Sigma=False, mu=dft_mu).copy()
        # calculate G(beta) via the function density, which is the same as fourier trafo G(w) and taking G(b)
        G_iw_lat_beta = G_iw_lat.density()
        # Doing the formula
        E_kin += np.trace(np.dot(H_ks[ik, 0, :nb, :nb], G_iw_lat_beta['up'][:, :]))
        E_kin += np.trace(np.dot(H_ks[ik, 0, :nb, :nb], G_iw_lat_beta['down'][:, :]))
    E_kin = float(E_kin.real)

    # collect data and put into E_kin_dft
    E_kin_dft = mpi.all_reduce(mpi.world, E_kin, lambda x, y: x+y)
    mpi.barrier()
    # E_kin should be divided by the number of k-points
    E_kin_dft = E_kin_dft/num_kpts

    if mpi.is_master_node():
        print('Kinetic energy contribution dft part: '+str(E_kin_dft))

    return E_kin_dft

def calc_bandcorr_man(general_parameters, sum_k, E_kin_dft):
    """
    Calculates the correlated kinetic energy from DMFT for target states
    and then determines the band correction energy

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict

    sum_k : SumK Object instances

    E_kin_dft: float
        kinetic energy from DFT


    __Returns:__
    E_bandcorr: float
        band energy correction E_kin_dmft - E_kin_dft

    """
    E_kin_dmft = 0.0j
    E_kin = 0.0j
    H_ks = sum_k.hopping
    num_kpts = sum_k.n_k

    # kinetic energy from dmft lattice Greens functions
    ikarray = np.array(range(sum_k.n_k))
    for ik in mpi.slice_array(ikarray):
        nb = int(sum_k.n_orbitals[ik])
        # calculate lattice greens function
        G_iw_lat = sum_k.lattice_gf(ik, beta=general_parameters['beta'], with_Sigma=True, with_dc=True).copy()
        # calculate G(beta) via the function density, which is the same as fourier trafo G(w) and taking G(b)
        G_iw_lat_beta = G_iw_lat.density()
        # Doing the formula
        E_kin += np.trace(np.dot(H_ks[ik, 0, :nb, :nb], G_iw_lat_beta['up'][:, :]))
        E_kin += np.trace(np.dot(H_ks[ik, 0, :nb, :nb], G_iw_lat_beta['down'][:, :]))
    E_kin = float(E_kin.real)

    # collect data and put into E_kin_dmft
    E_kin_dmft = mpi.all_reduce(mpi.world, E_kin, lambda x, y: x+y)
    mpi.barrier()
    # E_kin should be divided by the number of k-points
    E_kin_dmft = E_kin_dmft/num_kpts

    if mpi.is_master_node():
        print('Kinetic energy contribution dmft part: '+str(E_kin_dmft))

    E_bandcorr = E_kin_dmft - E_kin_dft

    return E_bandcorr
