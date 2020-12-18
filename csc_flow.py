"""
contains the CSC flow control functions
"""

import time
from timeit import default_timer as timer
import subprocess
import shlex
import os.path

# triqs
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from triqs_dft_tools.converters.vasp import VaspConverter
import triqs_dft_tools.converters.plovasp.converter as plo_converter
from triqs_dft_tools.converters.wannier90 import Wannier90Converter

import toolset
from dmft_cycle import dmft_cycle
import vasp_manager as vasp

_WANNIER_SEEDNAME = 'wannier90'

def _run_plo_converter(general_parameters):
    # Checks for plo file for projectors
    if not os.path.exists(general_parameters['plo_cfg']):
        print('*** Input PLO config file not found! '
              + 'I was looking for {} ***'.format(general_parameters['plo_cfg']))
        mpi.MPI.COMM_WORLD.Abort(1)

    # Runs plo converter
    plo_converter.generate_and_output_as_text(general_parameters['plo_cfg'], vasp_dir='./')
    # Writes new H(k) to h5 archive
    converter = VaspConverter(filename=general_parameters['seedname'])
    converter.convert_dft_input()

def _run_wannier_converter(dft_parameters):
    if (not os.path.exists(_WANNIER_SEEDNAME + '.win')
        or not os.path.exists(_WANNIER_SEEDNAME + '.inp')):
        print('*** Wannier input/converter config file not found! '
              + 'I was looking for {0}.win and {0}.inp ***'.format(_WANNIER_SEEDNAME))
        mpi.MPI.COMM_WORLD.Abort(1)

    # Runs wannier90 twice:
    # First preprocessing to write nnkp file, then normal run
    command = shlex.split(dft_parameters['wannier90_exec'])
    subprocess.check_call(command[:1] + ['-pp'] + command[1:], shell=False)
    subprocess.check_call(command, shell=False)
    # Writes new H(k) to h5 archive
    #TODO: choose rot_mat_type with general_parameters['set_rot']
    converter = Wannier90Converter(_WANNIER_SEEDNAME, rot_mat_type='hloc_diag', bloch_basis=True)
    converter.convert_dft_input()

# Main CSC flow method
def csc_flow_control(general_parameters, solver_parameters, dft_parameters, advanced_parameters):
    """
    function to run the csc cycle. It writes and removes the vasp.lock file to
    start and stop Vasp, run the converter, run the dmft cycle and abort the job
    if all iterations are finished.

    Parameters
    ----------
    general_parameters : dict
        general parameters as a dict
    solver_parameters : dict
        solver parameters as a dict
    dft_parameters : dict
        dft parameters as a dict
    advanced_parameters : dict
        advanced parameters as a dict

    __Returns:__
    nothing

    """

    vasp_process_id = vasp.start(dft_parameters['n_cores'], dft_parameters['executable'],
                                 dft_parameters['mpi_env'])

    mpi.report('  Waiting for VASP lock to appear...')
    while not vasp.is_lock_file_present():
        time.sleep(1)

    # if GAMMA file already exists, load it by doing extra DFT iterations
    if os.path.exists('GAMMA'):
        iter_dft = -8
    else:
        iter_dft = 0

    iter_dmft = 0
    while iter_dmft < general_parameters['n_iter_dmft']:
        start_time_dft = timer()
        mpi.report('  Waiting for VASP lock to disappear...')
        mpi.barrier()
        #waiting for vasp to finish
        while vasp.is_lock_file_present():
            time.sleep(1)

        # check if we should do a dmft iteration now
        iter_dft += 1
        if (iter_dft-1) % dft_parameters['n_iter'] != 0 or iter_dft < 0:
            vasp.reactivate()
            continue

        end_time_dft = timer()

        # Runs the converter
        if mpi.is_master_node():
            if dft_parameters['projector_type'] == 'plo':
                _run_plo_converter(general_parameters)

                if dft_parameters['store_eigenvals']:
                    toolset.store_dft_eigvals(path_to_h5=general_parameters['seedname']+'.h5', iteration=iter_dmft+1)
            elif dft_parameters['projector_type'] == 'w90':
                _run_wannier_converter(dft_parameters)

        mpi.barrier()

        if mpi.is_master_node():
            print('\n' + '='*80)
            print('DFT cycle took {:10.4f} seconds'.format(end_time_dft-start_time_dft))
            print('calling dmft_cycle')
            print('DMFT iteration {} / {}'.format(iter_dmft+1, general_parameters['n_iter_dmft']))
            print('='*80 + '\n')

        # if first iteration the h5 archive needs to be prepared
        if iter_dmft == 0:
            if mpi.is_master_node():
                # basic H5 archive checks and setup
                h5_archive = HDFArchive(general_parameters['seedname']+'.h5', 'a')
                if 'DMFT_results' not in h5_archive:
                    h5_archive.create_group('DMFT_results')
                if 'last_iter' in h5_archive['DMFT_results']:
                    prev_iterations = h5_archive['DMFT_results']['iteration_count']
                    print('previous iteration count of {} will be added to total number of iterations'.format(prev_iterations))
                    general_parameters['prev_iterations'] = prev_iterations
                else:
                    h5_archive['DMFT_results'].create_group('last_iter')
                    general_parameters['prev_iterations'] = 0
                if 'DMFT_input' not in h5_archive:
                    h5_archive.create_group('DMFT_input')
                    h5_archive['DMFT_input'].create_group('solver')

            general_parameters = mpi.bcast(general_parameters)

        if mpi.is_master_node():
            start_time_dmft = timer()

        ############################################################
        # run the dmft_cycle
        dmft_cycle(general_parameters, solver_parameters, advanced_parameters)
        ############################################################

        if iter_dmft == 0:
            iter_dmft += general_parameters['n_iter_dmft_first']
        else:
            iter_dmft += general_parameters['n_iter_dmft_per']

        if mpi.is_master_node():
            end_time_dmft = timer()
            print('\n' + '='*80)
            print('DMFT cycle took {:10.4f} seconds'.format(end_time_dmft-start_time_dmft))
            print('running VASP now')
            print('='*80 + '\n')
        #######

        # creates the lock file and Vasp will be unleashed
        vasp.reactivate()

    # stop if the maximum number of dmft iterations is reached
    if mpi.is_master_node():
        print('\n  Maximum number of iterations reached.')
        print('  Aborting VASP iterations...\n')
        vasp.kill(vasp_process_id)
