"""
This python script allows one to perform DFT+DMFT calculations with VASP
or with a pre-defined h5 archive (only one-shot) for
multiband/many-correlated-shells systems using the TRIQS package,
in combination with the CThyb solver and SumkDFT from DFT-tools.
triqs version 2.0 or higher is required

Written by Alexander Hampel, Sophie Beck
Materials Theory, ETH Zurich,
"""

# the future numpy (>1.15) is not fully compatible with triqs 2.0 atm
# suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings('ignore')

# system
import os
import sys
import shutil
from timeit import default_timer as timer

# triqs
import pytriqs.utility.mpi as mpi
from pytriqs.archive import HDFArchive

# own modules
from read_config import read_config
from observables import prep_observables
from dmft_cycle import dmft_cycle
from csc_flow import csc_flow_control


def main():
    """The main function for one-shot and charge-self-consistent calculations"""
    # timing information
    if mpi.is_master_node():
        global_start = timer()

    # reading configuration for calculation
    general_parameters = None
    solver_parameters = None
    dft_parameters = None
    advanced_parameters = None
    if mpi.is_master_node():
        if len(sys.argv) > 1:
            config_file_name = str(sys.argv[1])
        else:
            config_file_name = 'dmft_config.ini'
        print('Reading the config file ' + config_file_name)
        general_parameters, solver_parameters, dft_parameters, advanced_parameters = read_config(config_file_name)
        general_parameters['config_file'] = config_file_name

        print('-'*25 + '\nGeneral parameters:')
        for key, value in general_parameters.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))
        print('-'*25 + '\nSolver parameters:')
        for key, value in solver_parameters.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))
        print('-'*25 + '\nDFT parameters:')
        for key, value in dft_parameters.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))
        print('-'*25 + '\nAdvanced parameters, don\'t change them unless you know what you are doing:')
        for key, value in advanced_parameters.items():
            print('{0: <20} {1: <4}'.format(key, str(value)))

    general_parameters = mpi.bcast(general_parameters)
    solver_parameters = mpi.bcast(solver_parameters)
    dft_parameters = mpi.bcast(dft_parameters)
    advanced_parameters = mpi.bcast(advanced_parameters)

    # start CSC calculation if csc is set to true
    if general_parameters['csc']:

        # check if seedname is only one Value
        if len(general_parameters['seedname']) > 1:
            mpi.report('!!! WARNING !!!')
            mpi.report('CSC calculations can only be done for one set of file at a time')

        # some basic setup that needs to be done for CSC calculations
        general_parameters['seedname'] = general_parameters['seedname'][0]
        general_parameters['jobname'] = '.'
        general_parameters['previous_file'] = 'none'

        # run the whole machinery
        csc_flow_control(general_parameters, solver_parameters, dft_parameters, advanced_parameters)

    # do a one-shot calculation with given h5 archive
    else:
        # extract filenames and do a dmft iteration for every h5 archive given
        number_calculations = len(general_parameters['seedname'])
        filenames = general_parameters['seedname']
        foldernames = general_parameters['jobname']
        mpi.report('{} DMFT calculation will be made for the following files: {}'.format(number_calculations, filenames))

        # check for h5 file(s)
        if mpi.is_master_node():
            for file in filenames:
                if not os.path.exists(file+'.h5'):
                    mpi.report('*** Input h5 file(s) not found! I was looking for '+file+'.h5 ***')
                    mpi.MPI.COMM_WORLD.Abort(1)

        for i, file in enumerate(foldernames):
            general_parameters['seedname'] = filenames[i]
            general_parameters['jobname'] = foldernames[i]
            if i == 0:
                general_parameters['previous_file'] = 'none'
            else:
                previous_file = filenames[i-1]
                previous_folder = foldernames[i-1]
                general_parameters['previous_file'] = previous_folder+'/'+previous_file+'.h5'

            if mpi.is_master_node():
                # create output directory
                print('calculation is performed in subfolder: '+general_parameters['jobname'])
                if not os.path.exists(general_parameters['jobname']):
                    os.makedirs(general_parameters['jobname'])

                    # copy h5 archive and config file to created folder
                    shutil.copyfile(general_parameters['seedname']+'.h5',
                                    general_parameters['jobname']+'/'+general_parameters['seedname']+'.h5')
                    shutil.copyfile(general_parameters['config_file'],
                                    general_parameters['jobname']+'/'+general_parameters['config_file'])
                else:
                    print('#'*80+'\n WARNING! specified job folder already exists continuing previous job! \n'+'#'*80+'\n')

            mpi.report('#'*80)
            mpi.report('starting the DMFT calculation for '+str(general_parameters['seedname']))
            mpi.report('#'*80)

            # basic H5 archive checks and setup
            if mpi.is_master_node():
                h5_archive = HDFArchive(general_parameters['jobname']+'/'+general_parameters['seedname']+'.h5', 'a')
                if not 'DMFT_results' in h5_archive:
                    h5_archive.create_group('DMFT_results')
                if not 'last_iter' in h5_archive['DMFT_results']:
                    h5_archive['DMFT_results'].create_group('last_iter')
                if not 'DMFT_input' in h5_archive:
                    h5_archive.create_group('DMFT_input')

            # prepare observable dicts and files, which is stored on the master node
            observables = dict()
            if mpi.is_master_node():
                observables = prep_observables(general_parameters, h5_archive)
            observables = mpi.bcast(observables)

            ############################################################
            # run the dmft_cycle
            observables = dmft_cycle(general_parameters, solver_parameters, advanced_parameters, observables)
            ############################################################

    if mpi.is_master_node():
        global_end = timer()
        print('-------------------------------')
        print('overall elapsed time: %10.4f seconds'%(global_end-global_start))

if __name__ == '__main__':
    main()
