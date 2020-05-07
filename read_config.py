"""
Provides the read_config function to read the config file
Doc is created with https://github.com/freeman-lab/myopts
Using the command: myopts read_config.py -o doc.md -s false
"""

import configparser
import pytriqs.utility.mpi as mpi

def read_config(config_file):
    """
    Reads the config file (default dmft_config.ini). It consists out of 2 or 3
    sections, general, solver_parameters and optionally advanced_parameters.
    Comments are in general possible with with the delimiters ';' or
    '#'. However, this is only possible in the beginning of a line not within
    the line!

    For default values, the string 'none' is used. NoneType cannot be saved in
    an h5 archive (in the framework that we are using).

    Parameters
    ----------

    seedname : str or list of str
                seedname for h5 archive or for multiple if calculations should be connected
    jobname : str or list of str, optional, default=seedname
                one or multiple jobnames specifying the output directories
    csc : bool, optional, default=False
                are we doing a CSC calculation?
    plo_cfg : str, optional, default='plo.cfg'
                config file for PLOs for the converter
    h_int_type : int
                interaction type
                1 = density-density: currently only implemented for 5 orbitals
                2 = Kanamori: only defined for either the t2g or the eg subset
                3 = full Slater: currently not implemented
    U :  float or comma seperated list of floats
                U values for impurities if only one value is given, the same U is assumed for all impurities
    J :  float or comma seperated list of floats
                J values for impurities if only one value is given, the same J is assumed for all impurities
    beta : float
                inverse temperature for Greens function etc
    n_iter_dmft_first : int, optional, default = 10
                number of iterations in first dmft cycle to converge dmft solution
    n_iter_dmft_per : int, optional, default = 2
                number of iterations per dmft step in CSC calculations
    n_iter_dmft : int
            number of iterations per dmft cycle after first cycle
    dc_type : int
                0: FLL
                1: held formula, needs to be used with slater-kanamori h_int_type=2
                2: AMF
                3: FLL for eg orbitals only with U,J for Kanamori
    prec_mu : float
                general precision for determining the chemical potential at any time calc_mu is called
    dc_dmft : bool
                DC with DMFT occupation in each iteration -> True
                DC with DFT occupations after each DFT cycle -> False
    n_LegCoeff : int, needed if measure_g_l=True
                number of legendre coefficients
    h5_save_freq : int, optional default=5
                how often is the output saved to the h5 archive
    magnetic : bool, optional, default=False
                are we doing a magnetic calculations? If yes put magnetic to True
    magmom : list of float seperated by comma, optional default=[]
                init magnetic moments if magnetic is on. length must be #imps.
                This will be used as percentage factor for each imp in the initial
                self energy, pos sign for up (1+fac)*sigma, negative sign for down
                (1-fac)*sigma channel
    enforce_off_diag : bool, optional, default=False
                enforce off diagonal elements in block structure finder
    h_field : float, optional, default=0.0
                magnetic field
    sigma_mix : float, optional, default=1.0
                mixing sigma with previous iteration sigma for better convergency. 1.0 means no mixing
    dc : bool, optional, default=True
                dc correction on yes or no?
    calc_energies : bool, optional, default=True
                calc energies explicitly within the dmft loop
    block_threshold : float, optional, default=1e-05
                threshold for finding block structures in the input data (off-diag yes or no)
    spin_names : list of str, optional, default=up,down
                names for spin channels, usually no need to specifiy this
    load_sigma : bool, optional, default=False
                load a old sigma from h5 file
    path_to_sigma : str, needed if load_sigma is true
                path to h5 file from which the sigma should be loaded
    load_sigma_iter : int, optional, default= last iteration
                load the sigma from a specific iteration if wanted
    occ_conv_crit : float, optional, default= -1
                stop the calculation if a certain treshold is reached in the last occ_conv_it iterations
    occ_conv_it : int, optional, default= 10
                how many iterations should be considered for convergence?
    sampling_iterations : int, optional, default = 0
                for how many iterations should the solution sampled after the CSC loop is converged
    fixed_mu_value : float, optional, default = 'none'
                If given, the chemical potential remains fixed in calculations
    store_dft_eigenvals : bool, optional, default= False
                stores the dft eigenvals from LOCPROJ file in h5 archive
    afm_order : bool, optional, default=False
                copy self energies instead of solving explicitly for afm order
    set_rot : string, optional, default='none'
                Do NOT use this when your converter gives you a non-identity matrix.
                Therefore, can't generally be used with Wannier90.
                use density_mat_dft to diagonalize occupations = 'den'
                use hloc_dft to diagonalize occupations = 'hloc'
    oneshot_postproc_gamma_file : bool, optional, default=False
                write the GAMMA file for vasp after completed one-shot calculations
    measure_chi_SzSz: bool, optional, default=False
                measure the dynamic spin suszeptibility chi(sz,sz(tau))
                triqs.github.io/cthyb/unstable/guide/dynamic_susceptibility_notebook.html
    measure_chi_insertions: int, optional, default=100
                number of insertation for measurement of chi
    __Solver Parameters:__
    ----------
    length_cycle : int
                length of each cycle; number of sweeps before measurement is taken
    n_warmup_cycles : int
                number of warmup cycles before real measurement sets in
    n_cycles_tot : int
                total number of sweeps
    measure_g_l : bool
                measure Legendre Greens function
    n_LegCoeff : int
                number of Legendre coefficients
    max_time : int, optional, default=-1
                maximum amount the solver is allowed to spend in eacht iteration
    imag_threshold : float, optional, default= 10e-15
                threshold for imag part of G0_tau. be warned if symmetries are off in projection scheme imag parts can occur in G0_tau
    measure_density_matrix : bool, optional, default=False
                measures the impurity density matrix and sets also
                use_norm_as_weight to true
    measure_g_tau : bool,optional, default=True
                should the solver measure G(tau)?
    measure_pert_order: bool, optional, default=False
                measure perturbation order histograms: triqs.github.io/cthyb/latest/guide/perturbation_order_notebook.html
                stored in the h5 archive under DMFT_results per iteration stored in pert_order_imp_X and, pert_order_total_imp_X
    move_double : bool, optional, default=True
                double moves in solver
    perform_tail_fit : bool, optional, default=False
                tail fitting if legendre is off?
    fit_max_moment : int, optional
                max moment to be fitted
    fit_min_n : int, optional
                number of start matsubara frequency to start with
    fit_max_n : int, optional
                number of highest matsubara frequency to fit
    store_solver: bool, optional default = False
                store the whole solver object under DMFT_input in h5 archive
    random_seed: int, optional default by triqs
                if specified the int will be used for random seeds! Careful, this will give the same random
                numbers on all mpi ranks
    legendre_fit: bool, optional default = False
                filter noise of G(tau) with G_l, cutoff is taken from nLegCoeff

    __DFT code parameters (only for csc):__
    ----------
    n_cores : int
                number of cores for the DFT code (VASP)
    n_iter_dft : int, optional, default = 6
                number of dft iterations per cycle
    dft_executable : string, default= 'vasp_std'
                command for the DFT / VASP executable
    mpi_env : string, default= 'local'
                selection for mpi env for DFT / VASP in default this will only call VASP as mpirun -np n_cores_dft dft_executable

    __Advanced Parameters:__
    ----------
    dc_factor : float, optional, default = 'none' (corresponds to 1)
                If given, scales the dc energy by multiplying with this factor, usually < 1
    dc_fixed_value : float, optional, default = 'none'
                If given, it sets the DC (energy/imp) to this fixed value. Overwrites EVERY other DC configuration parameter if DC is turned on

    __Returns:__
    ----------
    general_parameters : dict

    solver_parameters : dict

    dft_parameters : dict

    advanced_parameters : dict

    """
    config = configparser.ConfigParser()
    config.read(config_file)

    general_parameters = {}
    solver_parameters = {}
    dft_parameters = {}
    advanced_parameters = {}

    # required parameters
    general_parameters['seedname'] = map(str, str(config['general']['seedname'].replace(' ', '')).split(','))
    general_parameters['h_int_type'] = int(config['general']['h_int_type'])
    general_parameters['U'] = map(float, str(config['general']['U']).split(','))
    general_parameters['J'] = map(float, str(config['general']['J']).split(','))
    general_parameters['beta'] = float(config['general']['beta'])
    general_parameters['n_iter_dmft'] = int(config['general']['n_iter_dmft'])
    general_parameters['dc_type'] = int(config['general']['dc_type'])
    general_parameters['prec_mu'] = float(config['general']['prec_mu'])
    general_parameters['dc_dmft'] = config['general'].getboolean('dc_dmft')

    # if csc we need the following input Parameters
    if 'csc' in config['general']:
        general_parameters['csc'] = config['general'].getboolean('csc')
    else:
        general_parameters['csc'] = False

    if general_parameters['csc']:
        if 'n_iter_dmft_first' in config['general']:
            general_parameters['n_iter_dmft_first'] = int(config['general']['n_iter_dmft_first'])
        else:
            general_parameters['n_iter_dmft_first'] = 10

        if 'n_iter_dmft_per' in config['general']:
            general_parameters['n_iter_dmft_per'] = int(config['general']['n_iter_dmft_per'])
        else:
            general_parameters['n_iter_dmft_per'] = 2

        if 'plo_cfg' in config['general']:
            general_parameters['plo_cfg'] = str(config['general']['plo_cfg'])
        else:
            general_parameters['plo_cfg'] = 'plo.cfg'

        if general_parameters['n_iter_dmft'] < general_parameters['n_iter_dmft_first']:
            mpi.report('*** error: total number of iterations should be at least = n_iter_dmft_first ***')
            mpi.MPI.COMM_WORLD.Abort(1)

        # DFT specific parameters
        dft_parameters['n_cores'] = int(config['dft']['n_cores'])

        if 'n_iter' in config['dft']:
            dft_parameters['n_iter'] = int(config['dft']['n_iter'])
        else:
            dft_parameters['n_iter'] = 6

        if 'executable' in config['dft']:
            dft_parameters['executable'] = str(config['dft']['executable'])
        else:
            dft_parameters['executable'] = 'vasp_std'

        if 'store_eigenvals' in config['dft']:
            dft_parameters['store_eigenvals'] = config['dft'].getboolean('store_eigenvals')
        else:
            dft_parameters['store_eigenvals'] = False

        if 'mpi_env' in config['dft']:
            dft_parameters['mpi_env'] = str(config['dft']['mpi_env'])
        else:
            dft_parameters['mpi_env'] = 'local'

    # optional stuff
    if 'jobname' in config['general']:
        general_parameters['jobname'] = map(str, str(config['general']['jobname'].replace(' ', '')).split(','))
        if len(general_parameters['jobname']) != len(general_parameters['seedname']):
            mpi.report('*** jobname must have same length as seedname. ***')
            mpi.MPI.COMM_WORLD.Abort(1)
    else:
        general_parameters['jobname'] = general_parameters['seedname']

    if 'h5_save_freq' in config['general']:
        general_parameters['h5_save_freq'] = int(config['general']['h5_save_freq'])
    else:
        general_parameters['h5_save_freq'] = 5

    if 'magnetic' in config['general']:
        general_parameters['magnetic'] = config['general'].getboolean('magnetic')
    else:
        general_parameters['magnetic'] = False

    if 'magmom' in config['general']:
        general_parameters['magmom'] = map(float, str(config['general']['magmom']).split(','))
    else:
        general_parameters['magmom'] = []

    if 'h_field' in config['general']:
        general_parameters['h_field'] = float(config['general']['h_field'])
    else:
        general_parameters['h_field'] = 0.0

    if 'sigma_mix' in config['general']:
        general_parameters['sigma_mix'] = float(config['general']['sigma_mix'])
    else:
        general_parameters['sigma_mix'] = 1.0

    if 'dc' in config['general']:
        general_parameters['dc'] = config['general'].getboolean('dc')
    else:
        general_parameters['dc'] = True

    if 'calc_energies' in config['general']:
        general_parameters['calc_energies'] = config['general'].getboolean('calc_energies')
    else:
        general_parameters['calc_energies'] = True

    if 'block_threshold' in config['general']:
        general_parameters['block_threshold'] = float(config['general']['block_threshold'])
    else:
        general_parameters['block_threshold'] = 1e-05

    if 'enforce_off_diag' in config['general']:
        general_parameters['enforce_off_diag'] = config['general'].getboolean('enforce_off_diag')
    else:
        general_parameters['enforce_off_diag'] = False

    if 'spin_names' in config['general']:
        general_parameters['spin_names'] = map(str, str(config['general']['spin_names']).split(','))
    else:
        general_parameters['spin_names'] = ['up', 'down']

    if 'load_sigma' in config['general']:
        general_parameters['load_sigma'] = config['general'].getboolean('load_sigma')
    else:
        general_parameters['load_sigma'] = False
    if general_parameters['load_sigma']:
        general_parameters['path_to_sigma'] = str(config['general']['path_to_sigma'])
    if 'load_sigma_iter' in config['general']:
        general_parameters['load_sigma_iter'] = int(config['general']['load_sigma_iter'])
    else:
        general_parameters['load_sigma_iter'] = -1

    if 'occ_conv_crit' in config['general']:
        general_parameters['occ_conv_crit'] = float(config['general']['occ_conv_crit'])
    else:
        general_parameters['occ_conv_crit'] = -1
    if 'occ_conv_it' in config['general']:
        general_parameters['occ_conv_it'] = int(config['general']['occ_conv_it'])
    else:
        general_parameters['occ_conv_it'] = 10

    if 'sampling_iterations' in config['general']:
        general_parameters['sampling_iterations'] = int(config['general']['sampling_iterations'])
    else:
        general_parameters['sampling_iterations'] = 0

    if 'fixed_mu_value' in config['general']:
        general_parameters['fixed_mu_value'] = float(config['general']['fixed_mu_value'])
    else:
        general_parameters['fixed_mu_value'] = 'none'
    if 'dft_mu' in config['general']:
        general_parameters['dft_mu'] = float(config['general']['dft_mu'])
    else:
        general_parameters['dft_mu'] = 0.0

    if 'store_dft_eigenvals' in config['general']:
        general_parameters['store_dft_eigenvals'] = config['general'].getboolean('store_dft_eigenvals')
    else:
        general_parameters['store_dft_eigenvals'] = False

    if 'afm_order' in config['general']:
        general_parameters['afm_order'] = config['general'].getboolean('afm_order')
    else:
        general_parameters['afm_order'] = False

    if 'set_rot' in config['general']:
        general_parameters['set_rot'] = str(config['general']['set_rot'])
    else:
        general_parameters['set_rot'] = 'none'

    if 'oneshot_postproc_gamma_file' in config['general']:
        general_parameters['oneshot_postproc_gamma_file'] = config['general'].getboolean('oneshot_postproc_gamma_file')
    else:
        general_parameters['oneshot_postproc_gamma_file'] = False

    if 'measure_chi_SzSz' in config['general']:
        general_parameters['measure_chi_SzSz'] = config['general'].getboolean('measure_chi_SzSz')
    else:
        general_parameters['measure_chi_SzSz'] = False
    if 'measure_chi_insertions' in config['general']:
        general_parameters['measure_chi_insertions'] = int(config['general']['measure_chi_insertions'])
    else:
        general_parameters['measure_chi_insertions'] = 100


    # solver related parameters
    # required parameters
    solver_parameters['length_cycle'] = int(config['solver_parameters']['length_cycle'])
    solver_parameters['n_warmup_cycles'] = int(config['solver_parameters']['n_warmup_cycles'])
    solver_parameters['n_cycles'] = int(float(config['solver_parameters']['n_cycles_tot'])) // mpi.size

    # optional stuff
    if 'max_time' in config['solver_parameters']:
        solver_parameters['max_time'] = int(config['solver_parameters']['max_time'])

    if 'imag_threshold' in config['solver_parameters']:
        solver_parameters['imag_threshold'] = float(config['solver_parameters']['imag_threshold'])

    if 'measure_g_tau' in config['solver_parameters']:
        solver_parameters['measure_G_tau'] = config['solver_parameters'].getboolean('measure_g_tau')
    else:
        solver_parameters['measure_G_tau'] = True

    if 'measure_density_matrix' in config['solver_parameters']:
        solver_parameters['measure_density_matrix'] = config['solver_parameters'].getboolean('measure_density_matrix')
        # also required to measure the density matrix
        solver_parameters['use_norm_as_weight'] = True
    else:
        solver_parameters['measure_density_matrix'] = False

    if 'move_double' in config['solver_parameters']:
        solver_parameters['move_double'] = config['solver_parameters'].getboolean('move_double')
    else:
        solver_parameters['move_double'] = True

    if 'measure_pert_order' in config['solver_parameters']:
        solver_parameters['measure_pert_order'] = config['solver_parameters'].getboolean('measure_pert_order')
    else:
        solver_parameters['measure_pert_order'] = False

    if 'move_shift' in config['solver_parameters']:
        solver_parameters['move_shift'] = config['solver_parameters'].getboolean('move_shift')
    else:
        solver_parameters['move_shift'] = True

    if 'random_seed' in config['solver_parameters']:
        solver_parameters['random_seed'] = int(config['solver_parameters']['random_seed'])


    if 'perform_tail_fit' in config['solver_parameters']:
        solver_parameters['perform_tail_fit'] = config['solver_parameters'].getboolean('perform_tail_fit')

        # if tailfit get parameters for fit
        if solver_parameters['perform_tail_fit']:
            if 'fit_max_moment' in config['solver_parameters']:
                solver_parameters['fit_max_moment'] = int(config['solver_parameters']['fit_max_moment'])
            if 'fit_min_n' in config['solver_parameters']:
                solver_parameters['fit_min_n'] = int(config['solver_parameters']['fit_min_n'])
            if 'fit_max_n' in config['solver_parameters']:
                solver_parameters['fit_max_n'] = int(config['solver_parameters']['fit_max_n'])
    else:
        solver_parameters['perform_tail_fit'] = False

    if 'measure_G_l' in config['solver_parameters']:
        solver_parameters['measure_G_l'] = config['solver_parameters'].getboolean('measure_G_l')

        if solver_parameters['measure_G_l']:
            # little workaround since #leg coefficients is not directly a solver parameter
            general_parameters['n_LegCoeff'] = int(config['solver_parameters']['n_LegCoeff'])
            # overwrite tail fitting!
            solver_parameters['perform_tail_fit'] = False

    if 'legendre_fit' in config['solver_parameters']:
        general_parameters['legendre_fit'] = config['solver_parameters'].getboolean('legendre_fit')

        # not compatible with tail fit or legendre sampling
        if general_parameters['legendre_fit'] and solver_parameters['measure_G_l']:
            print('\n Warning! legendre fit for Gtau can only be used without Gl measurement! Setting legendre_fit to false\n ')
            general_parameters['legendre_fit'] = False
        if general_parameters['legendre_fit'] and solver_parameters['perform_tail_fit']:
            print('\n Warning! legendre fit for Gtau can only be used without tail fitting! Setting legendre_fit to false\n ')
            general_parameters['legendre_fit'] = False

        # number of legendre coefficients
        if general_parameters['legendre_fit']:
            general_parameters['n_LegCoeff'] = int(config['solver_parameters']['n_LegCoeff'])

    else:
        general_parameters['legendre_fit'] = False

    if 'store_solver' in config['solver_parameters']:
        general_parameters['store_solver'] = config['solver_parameters'].getboolean('store_solver')
    else:
        general_parameters['store_solver'] = False

    # advanced parameters: non-standard DMFT settings
    if 'advanced_parameters' in config and 'dc_factor' in config['advanced_parameters']:
        advanced_parameters['dc_factor'] = float(config['advanced_parameters']['dc_factor'])
    else:
        advanced_parameters['dc_factor'] = 'none'

    if 'advanced_parameters' in config and 'dc_fixed_value' in config['advanced_parameters']:
        advanced_parameters['dc_fixed_value'] = float(config['advanced_parameters']['dc_fixed_value'])
    else:
        advanced_parameters['dc_fixed_value'] = 'none'


    return general_parameters, solver_parameters, dft_parameters, advanced_parameters
