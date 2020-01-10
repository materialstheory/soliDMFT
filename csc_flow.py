# contains the CSC flow control functions

import time
from timeit import default_timer as timer
import os.path
import socket
from collections import defaultdict

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

# Functions for interaction with VASP

def start_vasp_from_master_node(number_cores, vasp_command='vasp'):

    # get MPI env
    world = mpi.world
    vasp_pid = 0

    hostnames = world.gather(socket.gethostname(), root=0)
    if mpi.is_master_node():
        # create hostfile based on first number_cores ranks
        hostfile = 'vasp.hostfile'
        hosts = defaultdict(int)
        for h in hostnames[:number_cores]:
            hosts[h] += 1
        with open(hostfile, 'w') as f:
            for i in hosts.items():
                f.write("%s slots=%d \n"%i)

        # clean environment
        env = {}
        for e in ['PATH','LD_LIBRARY_PATH','SHELL','PWD','HOME',
                'OMP_NUM_THREADS','OMPI_MCA_btl_vader_single_copy_mechanism']:
            v = os.getenv(e)
            if v: env[e] = v

        # assuming that mpirun points to the correct mpi env
        exe = 'mpirun'
        for d in os.getenv('PATH', os.defpath).split(os.pathsep):
            if d:
                p = os.path.join(d, exe)
                if os.access(p, os.F_OK | os.X_OK):
                    exe = p
                    break

        # arguments for mpirun: for the scond node, mpirun starts VASP by using ssh, therefore we need to handover the env variables with -x
        args = [exe, '-hostfile', hostfile, '-np', str(number_cores),
                '-mca', 'mtl', '^psm2,ofi', '-x', 'LD_LIBRARY_PATH',
                '-x', 'PATH', '-x', 'OMP_NUM_THREADS', vasp_command]

        vasp_pid = os.fork()
        if vasp_pid == 0:
            # close_fds
            for fd in range(3,256):
                try:
                    os.close(fd)
                except OSError:
                    pass
            print("\n Starting VASP now \n")
            os.execve(exe, args, env)
            print('\n VASP exec failed \n')
            os._exit(127)

    mpi.barrier()
    vasp_pid = mpi.bcast(vasp_pid)

    return vasp_pid

def activate_vasp():
    if mpi.is_master_node():
        open('./vasp.lock', 'a').close()
    mpi.barrier()


# Main CSC flow method
def csc_flow_control(general_parameters, solver_parameters, dft_parameters):
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

    __Returns:__
    nothing

    """

    vasp_process = start_vasp_from_master_node(dft_parameters['n_cores'],
                                            dft_parameters['executable'])

    mpi.report("  Waiting for VASP lock to appear...")
    while not toolset.is_vasp_lock_present():
        time.sleep(1)

    # if GAMMA file already exists, load it by doing an extra DFT iteration
    if os.path.exists('GAMMA'):
        iter= -8
    else:
        iter = 0

    iter_dmft = 0
    start_time_dft = timer()
    while iter_dmft < general_parameters['n_iter_dmft']:

        mpi.report("  Waiting for VASP lock to disappear...")
        mpi.barrier()

        #waiting for vasp to finish
        while toolset.is_vasp_lock_present():
            time.sleep(1)

        # check if we should do a dmft iteration now
        iter += 1

        if (iter-1) % dft_parameters['n_iter'] != 0 or iter < 0:
            activate_vasp()
            continue

        # run the converter
        if mpi.is_master_node():
            end_time_dft = timer()

            # check for plo file for projectors
            if not os.path.exists(general_parameters['plo_cfg']):
                mpi.report('*** Input PLO config file not found! I was looking for '+str(general_parameters['plo_cfg'])+' ***')
                mpi.MPI.COMM_WORLD.Abort(1)

            # run plo converter
            plo_converter.generate_and_output_as_text(general_parameters['plo_cfg'], vasp_dir='./')

            # create h5 archive or build updated H(k)
            Converter = VaspConverter(filename=general_parameters['seedname'])

            # convert h5 archive now
            Converter.convert_dft_input()

            # if wanted store eigenvalues in h5 archive
            if dft_parameters['store_eigenvals']:
                toolset.store_dft_eigvals(config_file = general_parameters['plo_cfg'],
                                  path_to_h5 = general_parameters['seedname']+'.h5',
                                  iteration = iter_dmft+1)

        mpi.barrier()

        if mpi.is_master_node():
            print('')
            print("="*80)
            print('DFT cycle took %10.4f seconds'%(end_time_dft-start_time_dft))
            print("calling dmft_cycle")
            print("DMFT iteration"+str(iter_dmft+1)+"/"+str( general_parameters['n_iter_dmft']))
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
            start_time_dmft = timer()

        ############################################################
        # run the dmft_cycle
        observables = dmft_cycle(general_parameters, solver_parameters, observables)
        ############################################################

        if iter_dmft == 0:
            iter_dmft += general_parameters['n_iter_dmft_first']
        else:
            iter_dmft += general_parameters['n_iter_dmft_per']

        if mpi.is_master_node():
            end_time_dmft = timer()
            print('')
            print("="*80)
            print('DMFT cycle took %10.4f seconds'%(end_time_dmft-start_time_dmft))
            print("running VASP now")
            print("="*80)
            print('')
        #######

        # create the lock file and Vasp will be unleashed
        activate_vasp()

        # start the timer again for the dft loop
        if mpi.is_master_node():
            start_time_dft = timer()

    # stop if the maximum number of dmft iterations is reached
    if mpi.is_master_node():
        print("\n  Maximum number of iterations reached.")
        print("  Aborting VASP iterations...\n")
        with open('STOPCAR', 'wt') as f_stop:
            f_stop.write("LABORT = .TRUE.\n")
        os.kill(vasp_process, signal.SIGTERM)
        mpi.MPI.COMM_WORLD.Abort(1)
