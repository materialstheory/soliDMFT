# provides the read_config function to read the config file

import configparser as cp
import pytriqs.utility.mpi as mpi

def read_config(config_file):
    """
    Reads the config file (default dmft_config.ini) it consists out of 2
    sections. Comments are in general possible with with the delimiters ';' or
    '#'. However, this is only possible in the beginning of a line not within
    the line!

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
                interaction type # 1=dens-dens 2=kanamori 3=full-slater
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
    n_iter_dft : int, optional, default = 8
                number of dft iterations per cycle
    dc_type : int
                1: held formula, needs to be used with slater-kanamori h_int_type=2
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
    fixed_mu_value : float, optional default -> set fixed_mu = False
                fixed mu calculations
    store_dft_eigenvals : bool, optional, default= False
                stores the dft eigenvals from LOCPROJ file in h5 archive
    rm_complex : bool, optional, default=False
                removes the complex parts from G0 before the solver runs
    afm_order : bool, optional, default=False
                copy self energies instead of solving explicitly for afm order
    set_rot : string, optional, default='none'
                use density_mat_dft to diagonalize occupations = 'den'
                use hloc_dft to diagonalize occupations = 'hloc'

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
    max_time : int, optional, default=-1
                maximum amount the solver is allowed to spend in eacht iteration
    imag_threshold : float, optional, default= 10e-15
                thresold for imag part of G0_tau. be warned if symmetries are off in projection scheme imag parts can occur in G0_tau
    measure_g_tau : bool,optional, default=True
                should the solver measure G(tau)?
    move_double : bool, optional, default=True
                double moves in solver
    perform_tail_fit : bool, optional, default=False
                tail fitting if legendre is off?
    fit_max_moment : int, needed if perform_tail_fit = true
                max moment to be fitted
    fit_min_n : int, needed if perform_tail_fit = true
                number of start matsubara frequency to start with
    fit_max_n : int, needed if perform_tail_fit = true
                number of highest matsubara frequency to fit

    __Returns:__
    general_parameters : dict

    solver_parameters : dict

    """
    config = cp.ConfigParser()
    config.read(config_file)

    solver_parameters = {}
    general_parameters = {}

    # required parameters
    general_parameters['seedname'] = map(str, str(config['general']['seedname'].replace(" ","")).split(','))
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

        if 'n_iter_dmft_first' in config['general']:
            general_parameters['n_iter_dmft_first'] = int(config['general']['n_iter_dmft_first'])
        else:
            general_parameters['n_iter_dmft_first'] = 10

        if 'n_iter_dmft_per' in config['general']:
            general_parameters['n_iter_dmft_per'] = int(config['general']['n_iter_dmft_per'])
        else:
            general_parameters['n_iter_dmft_per'] = 2

        if 'n_iter_dft' in config['general']:
            general_parameters['n_iter_dft'] = int(config['general']['n_iter_dft'])
        else:
            general_parameters['n_iter_dft'] = 6

        if 'plo_cfg' in config['general']:
            general_parameters['plo_cfg'] = str(config['general']['plo_cfg'])
        else:
            general_parameters['plo_cfg'] = 'plo.cfg'

        if general_parameters['n_iter_dmft'] < general_parameters['n_iter_dmft_first']:
            mpi.report('*** error: total number of iterations should be at least = n_iter_dmft_first ***')
            mpi.MPI.COMM_WORLD.Abort(1)
    else:
        general_parameters['csc'] = False

    # optional stuff
    if 'jobname' in config['general']:
        general_parameters['jobname'] = map(str, str(config['general']['jobname'].replace(" ","")).split(','))
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
        general_parameters['spin_names'] = ['up','down']

    if 'load_sigma' in config['general']:
        general_parameters['load_sigma'] = config['general'].getboolean('load_sigma')
    else:
        general_parameters['load_sigma'] = False
    if general_parameters['load_sigma'] == True:
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
        general_parameters['fixed_mu'] = True
    else:
        general_parameters['fixed_mu'] = False
    if 'dft_mu' in config['general']:
        general_parameters['dft_mu'] = float(config['general']['dft_mu'])
    else:
        general_parameters['dft_mu'] = 0.0

    if 'store_dft_eigenvals' in config['general']:
        general_parameters['store_dft_eigenvals'] = config['general'].getboolean('store_dft_eigenvals')
    else:
        general_parameters['store_dft_eigenvals'] = False

    if 'rm_complex' in config['general']:
        general_parameters['rm_complex'] = config['general'].getboolean('rm_complex')
    else:
        general_parameters['rm_complex'] = False

    if 'afm_order' in config['general']:
        general_parameters['afm_order'] = config['general'].getboolean('afm_order')
    else:
        general_parameters['afm_order'] = False

    if 'set_rot' in config['general']:
        general_parameters['set_rot'] = str(config['general']['set_rot'])
    else:
        general_parameters['set_rot'] = 'none'

    # solver related parameters
    # required parameters
    solver_parameters["length_cycle"] = int(config['solver_parameters']['length_cycle'])
    solver_parameters["n_warmup_cycles"] = int(config['solver_parameters']['n_warmup_cycles'])
    solver_parameters["n_cycles"] = int( float(config['solver_parameters']['n_cycles_tot']) / (mpi.size))
    solver_parameters['measure_G_l'] = config['solver_parameters'].getboolean('measure_g_l')

    # optional stuff
    if 'max_time' in config['solver_parameters']:
        solver_parameters["max_time"] = int(config['solver_parameters']['max_time'])

    if 'imag_threshold' in config['solver_parameters']:
        solver_parameters["imag_threshold"] = float(config['solver_parameters']['imag_threshold'])

    if 'measure_g_tau' in config['solver_parameters']:
        solver_parameters["measure_G_tau"] = config['solver_parameters'].getboolean('measure_g_tau')
    else:
        solver_parameters["measure_G_tau"] = True

    if 'move_double' in config['solver_parameters']:
        solver_parameters["move_double"] = config['solver_parameters'].getboolean('move_double')
    else:
        solver_parameters["move_double"] = True

    #tailfitting only if legendre is off
    if solver_parameters['measure_G_l'] == True:
        # little workaround since #leg coefficients is not directly a solver parameter
        general_parameters["n_LegCoeff"] = int(config['solver_parameters']['n_LegCoeff'])
        solver_parameters["perform_tail_fit"] = False
    else:
        if 'perform_tail_fit' in config['solver_parameters']:
            solver_parameters["perform_tail_fit"] = config['solver_parameters'].getboolean('perform_tail_fit')
        else:
            solver_parameters["perform_tail_fit"] = False
    if solver_parameters["perform_tail_fit"] == True:
        solver_parameters["fit_max_moment"] = int(config['solver_parameters']['fit_max_moment'])
        solver_parameters["fit_min_n"] = int(config['solver_parameters']['fit_min_n'])
        solver_parameters["fit_max_n"] = int(config['solver_parameters']['fit_max_n'])

    return general_parameters,solver_parameters
