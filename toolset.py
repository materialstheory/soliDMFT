# contains all the helper functions:
# is_vasp_lock_present
# is_vasp_running
# store_dft_eigvals
# get_dft_energy
# get_dft_mu
# check_convergence
# determine_block_structure
# load_sigma_from_h5

import numpy as np

# triqs
from h5 import HDFArchive
from triqs_dft_tools.converters.plovasp.vaspio import VaspData
from triqs.operators.util.observables import S_op
from triqs.gf import BlockGf, GfImFreq
from triqs.gf.tools import fit_legendre
import triqs.utility.mpi as mpi

def store_dft_eigvals(path_to_h5, iteration):
    """
    save the eigenvalues from LOCPROJ file to calc directory
    """
    with HDFArchive(path_to_h5, 'a') as archive:
        if not 'dft_eigvals' in archive:
            archive.create_group('dft_eigvals')

        vasp_data = VaspData('./')
        eigenvals = vasp_data.plocar.eigs[:, :, 0]

        for ik in range(vasp_data.plocar.eigs[:, 0, 0].shape[0]):
            eigenvals[ik, :] = eigenvals[ik, :]-vasp_data.plocar.efermi

        archive['dft_eigvals']['it_'+str(iteration)] = eigenvals

def get_dft_energy():
    """
    Reads energy from the last line of OSZICAR.
    """
    with open('OSZICAR', 'r') as file:
        nextline = file.readline()
        while nextline.strip():
            line = nextline
            nextline = file.readline()
    try:
        dft_energy = float(line.split()[2])
    except ValueError:
        print('Cannot read energy from OSZICAR, setting it to zero')
        dft_energy = 0.0
    return dft_energy

# TODO: remove unused get_dft_mu?
def get_dft_mu():
    """
    Reads fermi energy from the first line of LOCPROJ.
    """
    with open('LOCPROJ', 'r') as file:
        line = file.readline()
    try:
        fermi_energy = float(line.split()[4])
    except ValueError:
        print('Cannot read energy from OSZICAR, setting it to zero')
        fermi_energy = 0.0
    return fermi_energy

def check_convergence(sum_k, general_parameters, observables):
    """
    check last x iterations for convergence and stop if criteria is reached

    Parameters
    ----------
    sum_k : SumK Object instances

    general_parameters : dict
        general parameters as a dict

    observables : list of dicts
        observable arrays

    __Returns:__
    converged : bool
        true if desired accuracy is reached

    std_dev : list of floats
        list of std_dev from the last #iterations

    """
    iterations = general_parameters['occ_conv_it']

    print('='*60)
    print('checking covergence of the last {} iterations:'.format(iterations))
    #loading the observables file
    avg_occ = np.empty(sum_k.n_inequiv_shells)
    std_dev = np.empty(sum_k.n_inequiv_shells)

    for icrsh in range(sum_k.n_inequiv_shells):
        imp_occ = np.sum([observables['imp_occ'][icrsh][spin][-iterations:]
                          for spin in sum_k.spin_block_names[sum_k.SO]], axis=0)
        avg_occ[icrsh] = np.mean(imp_occ)
        std_dev[icrsh] = np.std(imp_occ)

        print('Average occupation of impurity {}: {:10.5f}'.format(icrsh, avg_occ[icrsh]))
        print('Standard deviation of impurity {}: {:10.5f}'.format(icrsh, std_dev[icrsh]))

    print('='*60 + '\n')

    return np.all(std_dev < general_parameters['occ_conv_crit']), std_dev

def determine_block_structure(sum_k, general_parameters):
    """
    determines block structrure and degenerate deg_shells
    computes first DFT density matrix to determine block structure and changes
    the density matrix according to needs i.e. magnetic calculations, or keep
    off-diag elements

    Parameters
    ----------
    sum_k : SumK Object instances

    __Returns:__
    sum_k : SumK Object instances
        updated sum_k Object
    """
    mpi.report('\n *** determination of block structure ***')

    # this returns a list of dicts (one entry for each corr shell)
    # the dict contains one entry for up and one for down
    # each entry is a square complex numpy matrix with dim=corr_shell['dim']
    dens_mat = sum_k.density_matrix(method='using_gf', beta=general_parameters['beta'])

    # if we want to do a magnetic calculation we need to lift up/down degeneracy
    if not general_parameters['csc'] and general_parameters['magnetic'] and sum_k.SO == 0:
        mpi.report('magnetic calculation: removing the spin degeneracy from the block structure')
        for i, elem in enumerate(dens_mat):
            for key, value in elem.items():
                if key == 'up':
                    for a in range(len(value[:, 0])):
                        for b in range(len(value[0, :])):
                            if a == b:
                                dens_mat[i][key][a, b] = value[a, b]*1.1
                elif key == 'down':
                    for a in range(len(value[:, 0])):
                        for b in range(len(value[0, :])):
                            if a == b:
                                dens_mat[i][key][a, b] = value[a, b]*0.9
                else:
                    mpi.report('warning spin channels not found! Doing a PM calculation')

    # for certain systems it is needed to keep off diag elements
    # this enforces to use the full corr subspace matrix
    if general_parameters['enforce_off_diag']:
        mpi.report('enforcing off-diagonal elements in block structure finder')
        for dens_mat_per_imp in dens_mat:
            for dens_mat_per_block in dens_mat_per_imp.values():
                dens_mat_per_block += 2 * general_parameters['block_threshold']

    mpi.report('using 1-particle density matrix and Hloc (atomic levels) to '
               'determine the block structure')
    sum_k.analyse_block_structure(dm=dens_mat, threshold=general_parameters['block_threshold'])

    return sum_k

def print_block_sym(sum_k):
    # Summary of block structure finder and determination of shell_multiplicity
    if mpi.is_master_node():
        print('\n number of ineq. correlated shells: {}'.format(sum_k.n_inequiv_shells))
        # correlated shells and their structure
        print('\n block structure summary')
        for icrsh in range(sum_k.n_inequiv_shells):
            shlst = [ish for ish, ineq_shell in enumerate(sum_k.corr_to_inequiv) if ineq_shell == icrsh]
            print(' -- Shell type #{:3d}: '.format(icrsh) + format(shlst))
            print('  | shell multiplicity '+str(len(shlst)))
            print('  | block struct. : ' + format(sum_k.gf_struct_solver[icrsh]))
            print('  | deg. orbitals : ' + format(sum_k.deg_shells[icrsh]))

        # Prints matrices
        print('\nRotation matrices')
        for icrsh, rot_crsh in enumerate(sum_k.rot_mat):
            n_orb = sum_k.corr_shells[icrsh]['dim']
            print('rot_mat[{:2d}] '.format(icrsh)+'real part'.center(9*n_orb)+'  '+'imaginary part'.center(9*n_orb))
            fmt = '{:9.5f}' * n_orb
            for row in rot_crsh:
                row = np.concatenate((row.real, row.imag))
                print((' '*11 + fmt + '  ' + fmt).format(*row))
        print('\n')

def load_sigma_from_h5(h5_archive, iteration):
    """
    Reads impurity self-energy for all impurities from file and returns them as a list

    Parameters
    ----------
    h5_archive : HDFArchive
        HDFArchive to read from
    iteration : int
        at which iteration will sigma be loaded

    __Returns:__
    self_energies : list of green functions

    dc_imp : numpy array
        DC potentials
    dc_energy : numpy array
        DC energies per impurity
    density_matrix : numpy arrays
        Density matrix from the previous self-energy
    """

    internal_path = 'DMFT_results/'
    internal_path += 'last_iter' if iteration == -1 else 'it_{}'.format(iteration)

    n_inequiv_shells = h5_archive['dft_input']['n_inequiv_shells']

    # Loads previous self-energies and DC
    self_energies = [h5_archive[internal_path]['Sigma_iw_{}'.format(iineq)]
                     for iineq in range(n_inequiv_shells)]
    dc_imp = h5_archive[internal_path]['DC_pot']
    dc_energy = h5_archive[internal_path]['DC_energ']

    # Loads density_matrix to recalculate DC if dc_dmft
    density_matrix = h5_archive[internal_path]['dens_mat_post']

    print('Loaded Sigma_imp0...imp{} '.format(n_inequiv_shells-1)
          + ('at last it ' if iteration == -1 else 'at it {} '.format(iteration)))

    return self_energies, dc_imp, dc_energy, density_matrix


def sumk_sigma_to_solver_struct(sum_k, start_sigma):
    """
    Extracts the local Sigma. Copied from SumkDFT.extract_G_loc, version 2.1.x.

    Parameters
    ----------
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    start_sigma : list of BlockGf (Green's function) objects
        List of Sigmas in sum_k block structure that are to be converted.

    Returns
    -------
    Sigma_inequiv : list of BlockGf (Green's function) objects
        List of Sigmas that can be used to initialize the solver
    """

    Sigma_local = [start_sigma[icrsh].copy() for icrsh in range(sum_k.n_corr_shells)]
    Sigma_inequiv = [BlockGf(name_block_generator=[(block, GfImFreq(indices=inner, mesh=Sigma_local[0].mesh))
                                                   for block, inner in sum_k.gf_struct_solver[ish].items()],
                             make_copies=False) for ish in range(sum_k.n_inequiv_shells)]

    # G_loc is rotated to the local coordinate system
    if sum_k.use_rotations:
        for icrsh in range(sum_k.n_corr_shells):
            for bname, gf in Sigma_local[icrsh]:
                Sigma_local[icrsh][bname] << sum_k.rotloc(
                    icrsh, gf, direction='toLocal')

    # transform to CTQMC blocks
    for ish in range(sum_k.n_inequiv_shells):
        for block, inner in sum_k.gf_struct_solver[ish].items():
            for ind1 in inner:
                for ind2 in inner:
                    block_sumk, ind1_sumk = sum_k.solver_to_sumk[ish][(block, ind1)]
                    block_sumk, ind2_sumk = sum_k.solver_to_sumk[ish][(block, ind2)]
                    Sigma_inequiv[ish][block][ind1, ind2] << Sigma_local[
                        sum_k.inequiv_to_corr[ish]][block_sumk][ind1_sumk, ind2_sumk]

    # return only the inequivalent shells
    return Sigma_inequiv


def _round_to_int(data):
    return (np.array(data) + .5).astype(int)


def load_crpa_interaction_matrix(sum_k, filename='UIJKL'):
    """
    Loads VASP cRPA data to use as an interaction Hamiltonian.
    """
    # Loads data from VASP cRPA file
    data = np.loadtxt(filename, unpack=True)
    u_matrix_four_indices = np.zeros(_round_to_int(np.max(data[:4], axis=1)), dtype=complex)
    for entry in data.T:
        # VASP switches the order of the indices, ijkl -> ikjl
        i, k, j, l = _round_to_int(entry[:4])-1
        u_matrix_four_indices[i, j, k, l] = entry[4] + 1j * entry[5]

    # Slices up the four index U-matrix, separating shells
    u_matrix_four_indices_per_shell = [None] * sum_k.n_inequiv_shells
    first_index_shell = 0
    for ish in range(sum_k.n_corr_shells):
        n_orb = sum_k.corr_shells[ish]['dim']
        icrsh = sum_k.corr_to_inequiv[ish]
        u_matrix_temp = u_matrix_four_indices[first_index_shell:first_index_shell+n_orb,
                                              first_index_shell:first_index_shell+n_orb,
                                              first_index_shell:first_index_shell+n_orb,
                                              first_index_shell:first_index_shell+n_orb]

        if ish == icrsh:
            u_matrix_four_indices_per_shell[icrsh] = u_matrix_temp
        elif not np.allclose(u_matrix_four_indices_per_shell[icrsh], u_matrix_temp, atol=1e-6, rtol=0):
            # TODO: for some reason, some entries in the matrices differ by a sign. Check that
            print(np.allclose(np.abs(u_matrix_four_indices_per_shell[icrsh]), np.abs(u_matrix_temp),
                              atol=1e-6, rtol=0))
            print('Warning: cRPA matrix for impurity {} '.format(icrsh)
                  + 'differs for shells {} and {}'.format(sum_k.inequiv_to_corr[icrsh], ish))

        first_index_shell += n_orb

    if not np.allclose(u_matrix_four_indices.shape, first_index_shell):
        print('Warning: different number of orbitals in cRPA matrix than in calculation.')

    return u_matrix_four_indices_per_shell


def adapt_U_2index_for_SO(Umat, Upmat):
    """
    Changes the two-index U matrices such that for a system consisting of a
    single block 'ud' with the entries (1, up), (1, down), (2, up), (2, down),
    ... the matrices are consistent with the case without spin-orbit coupling.

    Parameters
    ----------
    Umat : numpy array
        The two-index interaction matrix for parallel spins without SO.
    Upmat : numpy array
        The two-index interaction matrix for antiparallel spins without SO.

    Returns
    -------
    Umat_SO : numpy array
        The two-index interaction matrix for parallel spins. Because in SO all
        entries have nominal spin 'ud', this matrix now contains the original
        Umat and Upmat.
    Upmat_SO : numpy array
        The two-index interaction matrix for antiparallel spins. Unused because
        in SO, all spins have the same nominal spin 'ud'.
    """

    Umat_SO = np.zeros(np.array(Umat.shape)*2, dtype=Umat.dtype)
    Umat_SO[::2, ::2] = Umat_SO[1::2, 1::2] = Umat
    Umat_SO[::2, 1::2] = Umat_SO[1::2, ::2] = Upmat
    Upmat_SO = None

    return Umat_SO, Upmat_SO


def adapt_U_4index_for_SO(Umat_full):
    """
    Changes the four-index U matrix such that for a system consisting of a
    single block 'ud' with the entries (1, up), (1, down), (2, up), (2, down),
    ... the matrix is consistent with the case without spin-orbit coupling.
    This can be derived directly from the definition of the Slater Hamiltonian.

    Parameters
    ----------
    Umat_full : numpy array
       The four-index interaction matrix without SO.

    Returns
    -------
    Umat_full_SO : numpy array
        The four-index interaction matrix with SO. For a matrix U_ijkl, the
        indices i, k correspond to spin sigma, and indices j, l to sigma'.
    """

    Umat_full_SO = np.zeros(np.array(Umat_full.shape)*2, dtype=Umat_full.dtype)
    for spin, spin_prime in ((0, 0), (0, 1), (1, 0), (1, 1)):
        Umat_full_SO[spin::2, spin_prime::2, spin::2, spin_prime::2] = Umat_full

    return Umat_full_SO


def legendre_filter(G_tau, order=100, G_l_cut=1e-19):
    """ Filter binned imaginary time Green's function
    using a Legendre filter of given order and coefficient threshold.

    Parameters
    ----------
    G_tau : TRIQS imaginary time Block Green's function
    auto : determines automatically the cut-off nl
    order : int
        Legendre expansion order in the filter
    G_l_cut : float
        Legendre coefficient cut-off
    Returns
    -------
    G_l : TRIQS Legendre Block Green's function
        Fitted Green's function on a Legendre mesh
    """

    # determine number of coefficients if auto=True
    # if auto:
        # print('determining number of legendre coefficients from decay! Check carefully!')
        # l_g_l_check = []

        # for b, g in G_tau:

            # # choose large order to find noise lvl
            # g_l = fit_legendre(g, order=100)
            # enforce_discontinuity(g_l, np.array([[1.]]))
            # l_g_l_check.append(g_l)

        # G_l_check = BlockGf(name_list=list(G_tau.indices), block_list=l_g_l_check)

        # nl_cut = []
        # # for each block
        # for blck, G_l_block in G_l_check:
            # n_orb = G_l_block.target_shape[0]
            # nl_even = len(G_l_block[0,0].data[0::2])
            # # loop over orbitals
            # for i_orb in range(0,n_orb):

                # # only take even Gls [0::2]
                # G_l_orb = np.abs(G_l_block[i_orb,i_orb].data[0::2])

                # # very simple determination, determine when
                # # decay stops from coefficents 8 on
                # for i_l in range(4,nl_even,1):
                    # if (G_l_orb[i_l] > G_l_orb[i_l-1]):
                        # nl_cut.append(2*i_l)
                        # break

        # order = int(np.average(nl_cut)+4)

        # print('orbitally averaged determined number of legendre coefficients: '+str(order))

    # final run with automatically determined number of coefficients or given order
    l_g_l = []

    for _, g in G_tau:

        g_l = fit_legendre(g, order=order)
        g_l.data[:] *= (np.abs(g_l.data) > G_l_cut)
        g_l.enforce_discontinuity(np.identity(g.target_shape[0]))

        l_g_l.append(g_l)

    G_l = BlockGf(name_list=list(G_tau.indices), block_list=l_g_l)

    return G_l

def chi_SzSz_setup(sum_k, general_parameters, solver_parameters):
    """

    Parameters
    ----------
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    general_paramters: general params dict
    solver_parameters: solver params dict

    Returns
    -------
    solver_parameters :  dict
        solver_paramters for the QMC solver
    Sz_list : list of S_op operators to measure per impurity
    """

    mpi.report('setting up Chi(Sz,Sz(tau)) measurement')

    Sz_list = [None] * sum_k.n_inequiv_shells

    for icrsh in range(sum_k.n_inequiv_shells):
        n_orb = sum_k.corr_shells[icrsh]['dim']
        orb_names = list(range(n_orb))

        Sz_list[icrsh] = S_op('z',
                              spin_names=sum_k.spin_block_names[sum_k.SO],
                              orb_names=orb_names,
                              map_operator_structure=sum_k.sumk_to_solver[icrsh])

    solver_parameters['measure_O_tau_min_ins'] = general_parameters['measure_chi_insertions']

    return solver_parameters, Sz_list
