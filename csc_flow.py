# contains the CSC flow control functions

import time
from timeit import default_timer as timer
import os

# triqs
import triqs.utility.mpi as mpi
from h5 import HDFArchive
from triqs_dft_tools.converters.vasp import VaspConverter
import triqs_dft_tools.converters.plovasp.converter as plo_converter


#from observables import *
from observables import prep_observables
import toolset
from dmft_cycle import dmft_cycle
import vasp_manager as vasp


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

    # if GAMMA file already exists, load it by doing an extra DFT iteration
    if os.path.exists('GAMMA'):
        iter = -8
    else:
        iter = 0

    iter_dmft = 0
    start_dft = timer()
    while iter_dmft < general_parameters['n_iter_dmft']:
        mpi.report('  Waiting for VASP lock to disappear...')
        mpi.barrier()
        #waiting for vasp to finish
        while vasp.is_lock_file_present():
            time.sleep(1)

        # check if we should do a dmft iteration now
        iter += 1
        if (iter-1) % dft_parameters['n_iter'] != 0 or iter < 0:
            vasp.reactivate()
            continue

        # run the converter
        if mpi.is_master_node():
            end_dft = timer()

            # check for plo file for projectors
            # Warning: if you plan to implement a W90 interface, make sure you
            # understand what happens in the orbital and what in KS basis
            if not os.path.exists(general_parameters['plo_cfg']):
                mpi.report('*** Input PLO config file not found! I was looking for '+str(general_parameters['plo_cfg'])+' ***')
                mpi.MPI.COMM_WORLD.Abort(1)

            # run plo converter
            plo_converter.generate_and_output_as_text(general_parameters['plo_cfg'], vasp_dir='./')

            # create h5 archive or build updated H(k)
            Converter = VaspConverter(filename=general_parameters['seedname'])

            # convert now h5 archive now
            Converter.convert_dft_input()

            # if wanted store eigenvalues in h5 archive
            if dft_parameters['store_eigenvals']:
                toolset.store_dft_eigvals(config_file=general_parameters['plo_cfg'], path_to_h5=general_parameters['seedname']+'.h5', iteration=iter_dmft+1)

        mpi.barrier()

        if mpi.is_master_node():
            print('\n' + '='*80)
            print('DFT cycle took {:10.4f} seconds'.format(end_dft-start_dft))
            print('calling dmft_cycle')
            print('DMFT iteration', iter_dmft+1, '/', general_parameters['n_iter_dmft'])
            print('='*80 + '\n')

        # if first iteration the h5 archive and observables need to be prepared
        if iter_dmft == 0:
            observables = dict()
            if mpi.is_master_node():
                # basic H5 archive checks and setup
                h5_archive = HDFArchive(general_parameters['seedname']+'.h5', 'a')
                if not 'DMFT_results' in h5_archive:
                    h5_archive.create_group('DMFT_results')
                if 'last_iter' in h5_archive['DMFT_results']:
                    prev_iterations = h5_archive['DMFT_results']['iteration_count']
                    print('previous iteration count of {} will be added to total number of iterations'.format(prev_iterations))
                    general_parameters['prev_iterations'] = prev_iterations
                else:
                    h5_archive['DMFT_results'].create_group('last_iter')
                    general_parameters['prev_iterations'] = 0
                if not 'DMFT_input' in h5_archive:
                    h5_archive.create_group('DMFT_input')
                    h5_archive['DMFT_input'].create_group('solver')

                # prepare observable dicts and files, which is stored on the master node

                observables = prep_observables(h5_archive)
            observables = mpi.bcast(observables)
            general_parameters = mpi.bcast(general_parameters)

        if mpi.is_master_node():
            start_dmft = timer()

        ############################################################
        # run the dmft_cycle
        observables = dmft_cycle(general_parameters, solver_parameters, advanced_parameters, observables)
        ############################################################

        if iter_dmft == 0:
            iter_dmft += general_parameters['n_iter_dmft_first']
        else:
            iter_dmft += general_parameters['n_iter_dmft_per']

        if mpi.is_master_node():
            end_dmft = timer()
            print('\n' + '='*80)
            print('DMFT cycle took {:10.4f} seconds'.format(end_dmft-start_dmft))
            print('running VASP now')
            print('='*80 + '\n')
        #######

        # creates the lock file and Vasp will be unleashed
        vasp.reactivate()

        # start the timer again for the dft loop
        if mpi.is_master_node():
            start_dft = timer()

    # stop if the maximum number of dmft iterations is reached
    if mpi.is_master_node():
        print('\n  Maximum number of iterations reached.')
        print('  Aborting VASP iterations...\n')
        vasp.kill(vasp_process_id)
