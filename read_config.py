"""
Provides the read_config function to read the config file

Reads the config file (default dmft_config.ini) with python's configparser
module. It consists of at least the section 'general', and optionally of the
sections 'solver', 'dft' and 'advanced'.
Comments are in general possible with with the delimiters ';' or
'#'. However, this is only possible in the beginning of a line not within
the line! For default values, the string 'none' is used. NoneType cannot be
saved in an h5 archive (in the framework that we are using).

List of all parameters, sorted by sections:

general
-------

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
h5_save_freq : int, optional, default=5
            how often is the output saved to the h5 archive
magnetic : bool, optional, default=False
            are we doing a magnetic calculations? If yes put magnetic to True.
            Not implemented for CSC calculations
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
noise_level_initial_sigma : float, optional, default=0.0
            spread of Gaussian noise applied to the initial Sigma
occ_conv_crit : float, optional, default= -1
            stop the calculation if a certain treshold is reached in the last occ_conv_it iterations
occ_conv_it : int, optional, default= 10
            how many iterations should be considered for convergence?
sampling_iterations : int, optional, default = 0
            for how many iterations should the solution sampled after the CSC loop is converged
sampling_h5_save_freq : int, optional, default = 5
            overwrites h5_save_freq when sampling has started
fixed_mu_value : float, optional, default = 'none'
            If given, the chemical potential remains fixed in calculations
mu_update_freq : int, optional, default = 1
            The chemical potential will be updated every # iteration
dft_mu : float, optional, default = 'none'
            The chemical potential of the DFT calculation.
            If not given, mu will be calculated from the DFT bands
mu_mix_const : float, optional, default = 1.0
            Constant term of the mixing of the chemical potential. See mu_mix_per_occupation_offset.
mu_mix_per_occupation_offset : float, optional, default = 0.0
            Mu mixing proportional to the occupation offset.
            Mixing between the dichotomy result and the previous mu,
            mu_next = factor * mu_dichotomy + (1-factor) * mu_previous, with
            factor = mu_mix_per_occupation_offset * |n - n_target| + mu_mix_const.
            The program ensures that 0 <= factor <= 1.
            mu_mix_const = 1.0 and mu_mix_per_occupation_offset = 0.0 means no mixing.
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

solver
------
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
fit_min_w : float, optional
            start matsubara frequency to start with
fit_max_w : float, optional
            highest matsubara frequency to fit
store_solver: bool, optional default = False
            store the whole solver object under DMFT_input in h5 archive
random_seed: int, optional default by triqs
            if specified the int will be used for random seeds! Careful, this will give the same random
            numbers on all mpi ranks
legendre_fit: bool, optional default = False
            filter noise of G(tau) with G_l, cutoff is taken from nLegCoeff

dft
---
n_cores : int
            number of cores for the DFT code (VASP)
n_iter : int, optional, default = 6
            number of dft iterations per cycle
executable : string, default= 'vasp_std'
            command for the DFT / VASP executable
mpi_env : string, default= 'local'
            selection for mpi env for DFT / VASP in default this will only call VASP as mpirun -np n_cores_dft dft_executable

advanced
--------
dc_factor : float, optional, default = 'none' (corresponds to 1)
            If given, scales the dc energy by multiplying with this factor, usually < 1
dc_fixed_value : float, optional, default = 'none'
            If given, it sets the DC (energy/imp) to this fixed value. Overwrites EVERY other DC configuration parameter if DC is turned on
dc_fixed_occ : list of float, optional, defaul = 'none'
            If given, the occupation for the DC for each impurity is set to the provided value.
            Still uses the same kind of DC!
dc_U :  float or comma seperated list of floats, optional, default = general_parameters['U']
            U values for DC determination if only one value is given, the same U is assumed for all impurities
dc_J :  float or comma seperated list of floats, optional, default = general_parameters['J']
            J values for DC determination if only one value is given, the same J is assumed for all impurities
"""

from configparser import ConfigParser
import pytriqs.utility.mpi as mpi
import numpy as np

# Workaround to get the default configparser boolean converter
BOOL_PARSER = lambda b: ConfigParser()._convert_to_boolean(b)

# TODO: it might be nicer to not have optional parameters at all and instead use
#       explicit default values

# Dictionary for the parameters. Contains the four sections general, dft, solver
# and advanced. Inside, all parameters are listed with their properties:
#   - converter: converter applied on the string value of the parameter
#   - valid for: a criterion for validity. If not fulfilled, the program crashes.
#                Always of form lambda x, params: ..., with x being the current parameter
#   - used: determines if parameter is used (and if not given, set to default value)
#           or unused and ignored. If 'used' and no default given, the program crashes.
#           If 'used' and default=None, this is an optional parameter
#   - default: default value for parameter. Can be a function of params but can only
#              use values that have NO default value. If it is None but 'used'
#              is True, the parameter becomes an optional parameter
PROPERTIES_PARAMS = {'general': {'seedname': {'converter': lambda s: s.replace(' ', '').split(','), 'used': True},

                                 'h_int_type': {'converter': int, 'valid for': lambda x, _: x in (1, 2, 3), 'used': True},

                                 'U': {'converter': lambda s: map(float, s.split(',')), 'used': True},

                                 'J': {'converter': lambda s: map(float, s.split(',')), 'used': True},

                                 'beta': {'converter': float, 'valid for': lambda x, _: x > 0, 'used': True},

                                 'n_iter_dmft': {'converter': int, 'valid for': lambda x, _: x >= 0, 'used': True},

                                 'dc': {'converter': BOOL_PARSER, 'used': True, 'default': True},

                                 'dc_type': {'converter': int, 'valid for': lambda x, _: x in (0, 1, 2, 3),
                                             'used': lambda params: params['general']['dc']},

                                 'prec_mu': {'converter': float, 'valid for': lambda x, _: x > 0, 'used': True},

                                 'dc_dmft': {'converter': BOOL_PARSER,
                                             'used': lambda params: params['general']['dc']},

                                 'csc': {'converter': BOOL_PARSER,
                                         'used': True, 'default': False},

                                 'n_iter_dmft_first': {'converter': int,
                                                       'valid for': lambda x, params: 0 < x <= params['general']['n_iter_dmft'],
                                                       'used': lambda params: params['general']['csc'], 'default': 10},

                                 'n_iter_dmft_per': {'converter': int, 'valid for': lambda x, _: x > 0,
                                                     'used': lambda params: params['general']['csc'], 'default': 2},

                                 'plo_cfg': {'used': lambda params: params['general']['csc'], 'default': 'plo.cfg'},

                                 'jobname': {'converter': lambda s: s.replace(' ', '').split(','),
                                             'valid for': lambda x, params: len(x) == len(params['general']['seedname']),
                                             'used': True, 'default': lambda params: params['general']['seedname']},

                                 'h5_save_freq': {'converter': int, 'valid for': lambda x, _: x > 0,
                                                  'used': True, 'default': 5},

                                 'magnetic': {'converter': BOOL_PARSER,
                                              'used': lambda params: not params['general']['csc'], 'default': False},

                                 # TODO: add check of length if possible
                                 'magmom': {'converter': lambda s: map(float, s.split(',')),
                                            'used': lambda params: not params['general']['csc'] and params['general']['magnetic'],
                                            'default': []},

                                 'h_field': {'converter': float, 'used': True, 'default': 0.0},

                                 'afm_order': {'converter': BOOL_PARSER,
                                               'used': lambda params: not params['general']['csc'] and params['general']['magnetic'],
                                               'default': False},

                                 'sigma_mix': {'converter': float, 'valid for': lambda x, _: x > 0,
                                               'used': True, 'default': 1.0},

                                 'calc_energies': {'converter': BOOL_PARSER, 'used': True, 'default': True},

                                 'block_threshold': {'converter': float, 'valid for': lambda x, _: x > 0,
                                                     'used': True, 'default': 1e-5},

                                 'enforce_off_diag': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                 'spin_names': {'converter': lambda s: s.replace(' ', '').split(','),
                                                'used': True, 'default': ['up', 'down']},

                                 'load_sigma': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                 'path_to_sigma': {'used': lambda params: params['general']['load_sigma']},

                                 'load_sigma_iter': {'converter': int,
                                                     'used': lambda params: params['general']['load_sigma'], 'default': -1},

                                 'noise_level_initial_sigma': {'converter': float,
                                                               'valid for': lambda x, _: x > 0 or np.isclose(x, 0),
                                                               'used': True, 'default': 0.},

                                 # TODO: change default to 'none'
                                 'occ_conv_crit': {'converter': float, 'used': True, 'default': -1},

                                 'occ_conv_it': {'converter': int, 'valid for': lambda x, _: x > 0,
                                                 'used': lambda params: params['general']['occ_conv_crit'] > 0, 'default': 10},

                                 'sampling_iterations': {'converter': int, 'valid for': lambda x, _: x >= 0,
                                                         'used': lambda params: params['general']['occ_conv_crit'] > 0,
                                                         'default': 0},

                                 'sampling_h5_save_freq': {'converter': int, 'valid for': lambda x, _: x > 0,
                                                           'used': lambda params: (params['general']['occ_conv_crit'] > 0
                                                                                   and params['general']['sampling_iterations'] > 0),
                                                           'default': 5},

                                 'fixed_mu_value': {'converter': float, 'used': True, 'default': 'none'},

                                 'mu_update_freq': {'converter': int, 'valid for': lambda x, _: x > 0,
                                                    'used': lambda params: params['general']['fixed_mu_value'] == 'none',
                                                    'default': 1},

                                 'dft_mu': {'converter': float, 'used': True, 'default': 'none'},

                                 'mu_mix_const': {'converter': float,
                                                  'valid for': lambda x, _: x > 0 or np.isclose(x, 0),
                                                  'used': lambda params: params['general']['fixed_mu_value'] == 'none',
                                                  'default': 1.},

                                 'mu_mix_per_occupation_offset': {'converter': float,
                                                                  'valid for': lambda x, _: x > 0 or np.isclose(x, 0),
                                                                  'used': lambda params: params['general']['fixed_mu_value'] == 'none',
                                                                  'default': 0.},

                                 # TODO: is this used?
                                 'store_dft_eigenvals': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                 'set_rot': {'valid for': lambda x, _: x in ('none', 'den', 'hloc'),
                                             'used': True, 'default': 'none'},

                                 'oneshot_postproc_gamma_file': {'converter': BOOL_PARSER,
                                                                 'used': lambda params: not params['general']['csc'], 'default': False},

                                 'measure_chi_SzSz': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                 'measure_chi_insertions': {'converter': int, 'used': True, 'default': 100},
                                },
                     'dft': {'n_cores': {'converter': int, 'valid for': lambda x, _: x > 0,
                                         'used': lambda params: params['general']['csc']},

                             'n_iter': {'converter': int, 'valid for': lambda x, _: x > 0,
                                        'used': lambda params: params['general']['csc'], 'default': 6},

                             'executable': {'used': lambda params: params['general']['csc'], 'default': 'vasp_std'},

                             'store_eigenvals': {'converter': BOOL_PARSER,
                                                 'used': lambda params: params['general']['csc'], 'default': False},

                             'mpi_env': {'valid for': lambda x, _: x in ('local', 'rusty', 'daint'),
                                         'used': lambda params: params['general']['csc'], 'default': 'local'},
                            },
                     'solver': {'length_cycle': {'converter': int, 'valid for': lambda x, _: x > 0, 'used': True},

                                'n_warmup_cycles': {'converter': int, 'valid for': lambda x, _: x > 0, 'used': True},

                                'n_cycles_tot': {'converter': lambda s: int(float(s)),
                                                 'valid for': lambda x, _: x >= 0, 'used': True},

                                'max_time': {'converter': int, 'valid for': lambda x, _: x >= 0,
                                             'used': True, 'default': None},

                                'imag_threshold': {'converter': float, 'used': True, 'default': None},

                                'measure_G_tau': {'converter': BOOL_PARSER, 'used': True, 'default': True},

                                'measure_density_matrix': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                'move_double': {'converter': BOOL_PARSER, 'used': True, 'default': True},

                                'measure_pert_order': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                'move_shift': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                'random_seed': {'converter': int, 'used': True, 'default': None},

                                'perform_tail_fit': {'converter': BOOL_PARSER,
                                                     'used': lambda params: not params['solver']['measure_G_l'],
                                                     'default': False},

                                'fit_max_moment': {'converter': int, 'valid for': lambda x, _: x >= 0,
                                                   'used': lambda params: 'perform_tail_fit' in params['solver']
                                                           and params['solver']['perform_tail_fit'], 'default': None},

                                'fit_min_n': {'converter': int, 'valid for': lambda x, _: x >= 0,
                                              'used': lambda params: 'perform_tail_fit' in params['solver']
                                                      and params['solver']['perform_tail_fit'], 'default': None},

                                'fit_max_n': {'converter': int, 'valid for': lambda x, params: x >= params['solver']['fit_min_n'],
                                              'used': lambda params: 'perform_tail_fit' in params['solver']
                                                      and params['solver']['perform_tail_fit'], 'default': None},

                                'fit_min_w': {'converter': float, 'valid for': lambda x, _: x >= 0,
                                              'used': lambda params: 'perform_tail_fit' in params['solver']
                                                      and params['solver']['perform_tail_fit'], 'default': None},

                                'fit_max_w': {'converter': float, 'valid for': lambda x, params: x >= params['solver']['fit_min_w'],
                                              'used': lambda params: 'perform_tail_fit' in params['solver']
                                                      and params['solver']['perform_tail_fit'], 'default': None},

                                'measure_G_l': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                'n_LegCoeff': {'converter': int, 'valid for': lambda x, _: x > 0,
                                               'used': lambda params: params['solver']['measure_G_l']
                                                       or params['solver']['legendre_fit']},

                                'legendre_fit': {'converter': BOOL_PARSER,
                                                 'used': lambda params: not params['solver']['measure_G_l']
                                                         and not params['solver']['perform_tail_fit'],
                                                 'default': False},

                                'store_solver': {'converter': BOOL_PARSER, 'used': True, 'default': False},
                               },
                     'advanced': {'dc_factor': {'converter': float, 'used': True, 'default': 'none'},

                                  'dc_fixed_value': {'converter': float, 'used': True, 'default': 'none'},

                                  'dc_fixed_occ': {'converter': lambda s: map(float, s.split(',')),
                                                   'used': True, 'default': 'none'},
                                  'dc_nominal': {'converter': BOOL_PARSER, 'used': True, 'default': False},

                                  'dc_U': {'converter': lambda s: map(float, s.split(',')),
                                           'used': True, 'default': lambda params: params['general']['U']},

                                  'dc_J': {'converter': lambda s: map(float, s.split(',')),
                                           'used': True, 'default': lambda params: params['general']['J']},
                                 }
                    }

# Mapping of {new name: legacy name}
LEGACY_SECTION_NAME_MAPPING = {'solver': 'solver_parameters', 'advanced': 'advanced_parameters'}


# -------------------------- config section cleanup --------------------------
def _config_find_default_section_entries(config):
    """
    Returns all items in the default section.

    Parameters
    ----------
    config : ConfigParser
        A configparser instance that has read the config file already.

    Returns
    -------
    list
        All entries in the default section.
    """
    return config['DEFAULT'].keys()

def _config_apply_sections_legacy_name_mapping(config):
    """
    Applies the mapping between legacy names of sections and new names. The
    mapping is saved in LEGACY_SECTION_NAME_MAPPING.

    Parameters
    ----------
    config : ConfigParser
        A configparser instance that has read the config file already.

    Returns
    -------
    config : ConfigParser
        The configparser where the section with legacy names are renamed.
    duplicate_sections : list
        All sections where the legacy name and the new name are found.
    renamed_sections : list
        All section where the legacy name was changed to the new name.
    """
    duplicate_sections = []
    renamed_sections = []

    for new_name, legacy_name in LEGACY_SECTION_NAME_MAPPING.items():
        # Only new name in there, everything is okay
        if new_name in config.keys() and legacy_name not in config.keys():
            continue

        # Both new section name and legacy name exists
        if new_name in config.keys() and legacy_name in config.keys():
            duplicate_sections.append((new_name, legacy_name))
            config.remove_section(legacy_name)
            continue

        # Only legacy name exists
        if new_name not in config.keys() and legacy_name in config.keys():
            renamed_sections.append((new_name, legacy_name))
            config.add_section(new_name)
            for param_name, param_value in config[legacy_name].items():
                config.set(new_name, param_name, param_value)
            config.remove_section(legacy_name)

    return config, duplicate_sections, renamed_sections

def _config_add_empty_sections(config):
    """
    Adds empty sections if no parameters in the whole section were given.

    Parameters
    ----------
    config : ConfigParser
        A configparser instance that has read the config file already.

    Returns
    -------
    config : ConfigParser
        The config parser with all required sections.
    """
    for section_name in PROPERTIES_PARAMS:
        if section_name not in config:
            config.add_section(section_name)

    return config

def _config_remove_unused_sections(config):
    """
    Removes sections that are not supported by this program.

    Parameters
    ----------
    config : ConfigParser
        A configparser instance that has read the config file already.

    Returns
    -------
    config : ConfigParser
        The config parser without unnecessary sections.
    unused_sections : list
        All sections that are not supported.
    """
    unused_sections = []
    for section_name in config.keys():
        if section_name != 'DEFAULT' and section_name not in PROPERTIES_PARAMS.keys():
            unused_sections.append(section_name)
            config.remove_section(section_name)

    return config, unused_sections

# -------------------------- parameter reading --------------------------
def _convert_parameters(config):
    """
    Applies the converter given in the PROPERTIES_PARAMS to the config. If no
    converter is given, a default string conversion is used.

    Parameters
    ----------
    config : ConfigParser
        A configparser instance that has passed through the above clean-up
        methods.

    Returns
    -------
    parameters : dict
        Contains dicts for each section. These dicts contain all parameter
        names and their respective value that are in the configparser and in
        the PROPERTIES_PARAMS.
    """
    parameters = {name: {} for name in PROPERTIES_PARAMS}

    for section_name, section_parameters in parameters.items():
        for param_name, param_props in PROPERTIES_PARAMS[section_name].items():
            if param_name not in config[section_name]:
                continue

            # Uses converter for parameters
            if 'converter' in param_props:
                section_parameters[param_name] = param_props['converter'](str(config[section_name][param_name]))
            else:
                section_parameters[param_name] = str(config[section_name][param_name])

    return parameters


def _find_nonexistent_parameters(config):
    """
    Returns all parameters that are in the config but not in the
    PROPERTIES_PARAMS and are therefore not recognized by the program.

    Parameters
    ----------
    config : ConfigParser
        A configparser instance that has passed through the above clean-up
        methods.

    Returns
    -------
    nonexistent_parameters : dict
        Contains a list for each section, which contains all unused parameter
        names.
    """
    nonexistent_parameters = {section_name: [] for section_name in PROPERTIES_PARAMS}

    for section_name, section_parameters in PROPERTIES_PARAMS.items():
        for param_name in config[section_name]:
            if param_name not in (key.lower() for key in section_parameters):
                nonexistent_parameters[section_name].append(param_name)

    return nonexistent_parameters


def _apply_default_values(parameters):
    """
    Applies default values to all parameters that were not given in the config.

    Parameters
    ----------
    parameters : dict
        Contains dicts for each section, which contain the parameter names and
        their values as read from the config file.

    Returns
    -------
    parameters : dict
        The parameters dict including the default values.
    default_values_used : dict
        Contains a list for each section, which contains all parameters that
        were set to their default values. Used to find out later which
        unnecessary parameters were actually given in the config file.
    """
    default_values_used = {section_name: [] for section_name in PROPERTIES_PARAMS}

    for section_name, section_parameters in PROPERTIES_PARAMS.items():
        for param_name, param_props in section_parameters.items():
            if 'default' not in param_props:
                continue

            if param_name in parameters[section_name]:
                continue

            default_values_used[section_name].append(param_name)
            if callable(param_props['default']):
                parameters[section_name][param_name] = param_props['default'](parameters)
            else:
                parameters[section_name][param_name] = param_props['default']

    return parameters, default_values_used


def _check_if_parameters_used(parameters, default_values_used):
    """
    Checks if the parameters in the config file are used or unnecessary.

    Parameters
    ----------
    parameters : dict
        Contains dicts for each section, which contain the parameter names and
        their values as read from the config file or otherwise set to default.
    default_values_used : dict
        Contains a list for each section, which contains all parameters that
        were set to their default values.

    Returns
    -------
    parameters : dict
        The parameters dict where all unnecessary parameters were removed.
    unnecessary_parameters : dict
        Contains a list for each section, which contains all parameters that
        were given in the config file but are unnecessary.
    missing_required_parameters : dict
        Contains a list for each section, which contains all parameters that
        are required, have no default and are missing from the config file.
    """
    unnecessary_parameters = {section_name: [] for section_name in PROPERTIES_PARAMS}
    missing_required_parameters = {section_name: [] for section_name in PROPERTIES_PARAMS}

    for section_name, section_parameters in PROPERTIES_PARAMS.items():
        for param_name, param_props in section_parameters.items():
            # 'used' could be bool or function returning bool
            if callable(param_props['used']):
                required = param_props['used'](parameters)
            else:
                required = param_props['used']

            if required:
                if param_name not in parameters[section_name]:
                    missing_required_parameters[section_name].append(param_name)
                elif parameters[section_name][param_name] is None:
                    del parameters[section_name][param_name]
                continue

            if param_name in parameters[section_name]:
                del parameters[section_name][param_name]

                if param_name not in default_values_used[section_name]:
                    unnecessary_parameters[section_name].append(param_name)

    return parameters, unnecessary_parameters, missing_required_parameters

def _checks_validity_criterion(parameters):
    """
    Checks the validity criterion from the PROPERTIES_PARAMS.

    Parameters
    ----------
    parameters : dict
        Contains dicts for each section, which contain the parameter names and
        their values as read from the config file, set to default if required
        or removed if they are unnecessary.

    Returns
    -------
    invalid_parameters : dict
        Contains a list for each section, which contains all parameters that
        do not fulfill their validity criterion.
    """
    invalid_parameters = {section_name: [] for section_name in PROPERTIES_PARAMS}

    for section_name, section_parameters in PROPERTIES_PARAMS.items():
        for name, value in parameters[section_name].items():
            if 'valid for' not in section_parameters[name]:
                continue

            condition = section_parameters[name]['valid for']
            if not condition(value, parameters):
                invalid_parameters[section_name].append(name)

    return invalid_parameters


def read_config(config_file):
    """
    Reads in the config file, checks its sections and parameters and returns
    the parameters sorted by their categories.

    Parameters
    ----------
    config_file : string
        File name of the config file usable for configparser.

    Raises
    ------
    ValueError
        Required parameters are missing or parameters do not fulfill their
        validity criterion.

    Returns
    -------
    general_parameters : dict

    solver_parameters : dict

    dft_parameters : dict

    advanced_parameters : dict
    """
    config = ConfigParser()
    config.read(config_file)

    # Checks if default section is empty
    config_default_entries = _config_find_default_section_entries(config)
    if config_default_entries:
        print('Warning: the following parameters are not in any section and will be ignored:')
        print(', '.join(config_default_entries))

    # Applies mapping of legacy names for sections and prints warnings
    config, duplicate_sections, renamed_sections = _config_apply_sections_legacy_name_mapping(config)
    if duplicate_sections:
        for new_name, legacy_name in duplicate_sections:
            print('Warning: the section "{0}" and its legacy "{1}" exist. "{1}" will be ignored'.format(new_name, legacy_name))

    if renamed_sections:
        for new_name, legacy_name in renamed_sections:
            print('Warning: old section name "{}" deprecated, please change it to "{}".'.format(legacy_name, new_name))

    # Adds empty sections if they don't exist
    config = _config_add_empty_sections(config)

    # Removes unused sections and prints a warning
    config, unused_sections = _config_remove_unused_sections(config)
    if unused_sections:
        print('Warning: ignoring parameters in following unexpected sections:')
        print(', '.join(unused_sections))

    # Reads and converts all valid parameters
    parameters = _convert_parameters(config)

    # Checks for unused parameters given in the config file
    nonexistent_parameters = _find_nonexistent_parameters(config)
    if any(nonexistent_parameters.values()):
        print('Warning: the following parameters are not supported:')
        for section_name, section_parameters in nonexistent_parameters.items():
            if section_parameters:
                print('- Section "{}": '.format(section_name) + ', '.join(section_parameters))

    # Applies default values
    parameters, default_values_used = _apply_default_values(parameters)

    # Prints warning if unnecessary parameters are given
    parameters, unnecessary_parameters, missing_required_parameters = _check_if_parameters_used(parameters, default_values_used)
    if any(unnecessary_parameters.values()):
        print('Warning: the following parameters are given but not used in this calculation:')
        for section_name, section_parameters in unnecessary_parameters.items():
            if section_parameters:
                print('- Section "{}": '.format(section_name) + ', '.join(section_parameters))

    # Raises error if required parameters are not given
    if any(missing_required_parameters.values()):
        required_error_string = ''
        for section_name, section_parameters in missing_required_parameters.items():
            if section_parameters:
                required_error_string += '\n- Section "{}": '.format(section_name) + ', '.join(section_parameters)
        raise ValueError('The following parameters are required but not given:'
                         + required_error_string)

    # Raises error if parameters invalid
    invalid_parameters = _checks_validity_criterion(parameters)
    if any(invalid_parameters.values()):
        invalid_error_string = ''
        for section_name, section_parameters in invalid_parameters.items():
            if section_parameters:
                invalid_error_string += '\n- Section "{}": '.format(section_name) + ', '.join(section_parameters)
        raise ValueError('The following parameters are not valid:'
                         + invalid_error_string)


    # Workarounds for some parameters
    parameters['solver']['n_cycles'] = parameters['solver']['n_cycles_tot'] // mpi.size
    del parameters['solver']['n_cycles_tot']

    if parameters['solver']['measure_density_matrix']:
        # also required to measure the density matrix
        parameters['solver']['use_norm_as_weight'] = True

    # little workaround since #leg coefficients is not directly a solver parameter
    if 'n_LegCoeff' in parameters['solver']:
        parameters['general']['n_LegCoeff'] = parameters['solver']['n_LegCoeff']
        del parameters['solver']['n_LegCoeff']

    # little workaround since #leg coefficients is not directly a solver parameter
    if 'legendre_fit' in parameters['solver']:
        parameters['general']['legendre_fit'] = parameters['solver']['legendre_fit']
        del parameters['solver']['legendre_fit']

    parameters['general']['store_solver'] = parameters['solver']['store_solver']
    del parameters['solver']['store_solver']

    return parameters['general'], parameters['solver'], parameters['dft'], parameters['advanced']
