# contains the CSC flow control functions

import time
from timeit import default_timer as timer
import os.path

# triqs
import pytriqs.utility.mpi as mpi
from pytriqs.archive import HDFArchive
try:
    # TRIQS 2.0
    from triqs_dft_tools.converters.vasp_converter import VaspConverter
    import triqs_dft_tools.converters.plovasp.converter as plo_converter
except ImportError:
    # TRIQS 1.4
    from pytriqs.applications.dft.sumk_dft import *
    from pytriqs.applications.dft.sumk_dft_tools import *
    from pytriqs.applications.dft.converters.vasp_converter import *
    import pytriqs.applications.dft.converters.plovasp.converter as plo_converter
    pass

#from observables import *
from observables import prep_observables
import toolset
from dmft_cycle import dmft_cycle

def csc_flow_control(general_parameters, solver_parameters):
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

    __Returns:__
    nothing

    """
    mpi.report("  Waiting for VASP lock to appear...")
    while not toolset.is_vasp_lock_present():
        time.sleep(1)

    # if GAMMA file already exists, load it by doing an extra DFT iteration
    if os.path.exists('GAMMA'):
        iter= -8
    else:
        iter = 0

    iter_dmft = 0
    start_dft = timer()
    while iter_dmft < general_parameters['n_iter_dmft']:

        mpi.report("  Waiting for VASP lock to disappear...")
        mpi.barrier()
        #waiting for vasp to finish
        while toolset.is_vasp_lock_present():
            time.sleep(1)

        # check if we should do a dmft iteration now
        iter += 1
        if (iter-1) % general_parameters['n_iter_dft'] != 0 or iter < 0:
            if mpi.is_master_node():
                open('./vasp.lock', 'a').close()
            continue

        # run the converter
        if mpi.is_master_node():
            end_dft = timer()

            # check for plo file for projectors
            if not os.path.exists(general_parameters['plo_cfg']):
                mpi.report('*** Input PLO config file not found! I was looking for '+str(general_parameters['plo_cfg'])+' ***')
                mpi.MPI.COMM_WORLD.Abort(1)

            # run plo converter
            plo_converter.generate_and_output_as_text(general_parameters['plo_cfg'], vasp_dir='./')


            # # create h5 archive or build updated H(k)

            Converter = VaspConverter(filename=general_parameters['seedname'])

            # convert now h5 archive now
            Converter.convert_dft_input()

            # if wanted store eigenvalues in h5 archive
            if general_parameters['store_dft_eigenvals']:
                toolset.store_dft_eigvals(config_file = general_parameters['plo_cfg'],
                                  path_to_h5 = general_parameters['seedname']+'.h5',
                                  iteration = iter_dmft+1)

        mpi.barrier()

        if mpi.is_master_node():
            print('')
            print("="*80)
            print('DFT cycle took %10.4f seconds'%(end_dft-start_dft))
            print("calling dmft_cycle")
            print("DMFT iteration", iter_dmft+1,"/", general_parameters['n_iter_dmft'])
            print("="*80)
            print('')

        # if first iteration the h5 archive and observables need to be prepared
        if iter_dmft == 0:
            observables = dict()
            if mpi.is_master_node():
                # basic H5 archive checks and setup
                h5_archive = HDFArchive(general_parameters['seedname']+'.h5','a')
                if not 'DMFT_results' in h5_archive:
                    h5_archive.create_group('DMFT_results')
                if not 'last_iter' in h5_archive['DMFT_results']:
                    h5_archive['DMFT_results'].create_group('last_iter')
                if not 'DMFT_input' in h5_archive:
                    h5_archive.create_group('DMFT_input')

                # prepare observable dicts and files, which is stored on the master node

                observables = prep_observables(general_parameters, h5_archive)
            observables = mpi.bcast(observables)

        if mpi.is_master_node():
            start_dmft = timer()

        ############################################################
        # run the dmft_cycle
        observables = dmft_cycle(general_parameters, solver_parameters, observables)
        ############################################################

        if iter_dmft == 0:
            iter_dmft += general_parameters['n_iter_dmft_first']
        else:
            iter_dmft += general_parameters['n_iter_dmft_per']

        if mpi.is_master_node():
            end_dmft = timer()
            print('')
            print("="*80)
            print('DMFT cycle took %10.4f seconds'%(end_dmft-start_dmft))
            print("running VASP now")
            print("="*80)
            print('')
        #######

        # remove the lock file and Vasp will be unleashed
        if mpi.is_master_node():
            open('./vasp.lock', 'a').close()

        # start the timer again for the dft loop
        if mpi.is_master_node():
            start_dft = timer()

    # stop if the maximum number of dmft iterations is reached
    if mpi.is_master_node():
        print("\n  Maximum number of iterations reached.")
        print("  Aborting VASP iterations...\n")
        f_stop = open('STOPCAR', 'wt')
        f_stop.write("LABORT = .TRUE.\n")
        f_stop.close()
        mpi.MPI.COMM_WORLD.Abort(1)
