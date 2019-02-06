# script to extract onsite density matrices and use them to construct h5 archive with vasp dft_tools interface for nickel 2 band eg model

# load needed modules
import numpy as np

import os.path
import shutil
import re
import cmath

from triqs_dft_tools.sumk_dft import *
from triqs_dft_tools.sumk_dft_tools import *
from triqs_dft_tools.converters.vasp_converter import *
from triqs_dft_tools.converters.plovasp.vaspio import VaspData
from triqs_dft_tools.converters.plovasp.plotools import generate_plo, output_as_text
import triqs_dft_tools.converters.plovasp.converter as plo_converter
from triqs_dft_tools.converters.plovasp.elstruct import ElectronicStructure

def calc_density_matrix(el_struct):
    """
    Calculate and output the density and overlap matrix out of projectors defined in el_struct.
    """
    plo = el_struct.proj_raw
    nproj, ns, nk, nb = plo.shape
    ions = sorted(list(set([param['isite'] for param in el_struct.proj_params])))
    nions = len(ions)
    norb = nproj / nions
    den_mat_sites = []
# Spin factor
    sp_fac = 2.0 if ns == 1 and not el_struct.nc_flag else 1.0

    den_mat = np.zeros((ns, nproj, nproj), dtype=np.float64)
    overlap = np.zeros((ns, nproj, nproj), dtype=np.float64)
#        ov_min = np.ones((ns, nproj, nproj), dtype=np.float64) * 100.0
#        ov_max = np.zeros((ns, nproj, nproj), dtype=np.float64)
    for ispin in xrange(ns):
        for ik in xrange(nk):
            kweight = el_struct.kmesh['kweights'][ik]
            occ = el_struct.ferw[ispin, ik, :]
            den_mat[ispin, :, :] += np.dot(plo[:, ispin, ik, :] * occ, plo[:, ispin, ik, :].T.conj()).real * kweight * sp_fac
            ov = np.dot(plo[:, ispin, ik, :], plo[:, ispin, ik, :].T.conj()).real
            overlap[ispin, :, :] += ov * kweight
#                ov_max = np.maximum(ov, ov_max)
#                ov_min = np.minimum(ov, ov_min)

# Output only the site-diagonal parts of the matrices
    for ispin in xrange(ns):
        print
        print "  Spin:", ispin + 1
        print ions
        for io, ion in enumerate(ions):
            print "  Site:", ion
            iorb_inds = [(ip, param['m']) for ip, param in enumerate(el_struct.proj_params) if param['isite'] == ion]
            norb = len(iorb_inds)
            dm = np.zeros((norb, norb))
            ov = np.zeros((norb, norb))
            for ind, iorb in iorb_inds:
                for ind2, iorb2 in iorb_inds:
                    dm[iorb, iorb2] = den_mat[ispin, ind, ind2]
                    ov[iorb, iorb2] = overlap[ispin, ind, ind2]
            den_mat_sites.append(dm)

            print "  Density matrix" + (12*norb - 12)*" " + "Overlap"
            for drow, dov in zip(dm, ov):
                out = ''.join(map("{0:12.7f}".format, drow))
                out += "    "
                out += ''.join(map("{0:12.7f}".format, dov))
                print out
    return den_mat_sites


vasp_dir = './'

# extract density matrices and write them to rotations file -> done
vasp_data = VaspData(vasp_dir, efermi_required=True)
el_struct = ElectronicStructure(vasp_data)
den_mat = calc_density_matrix(el_struct)

rot_file = open('rotations.dat','w')
print den_mat
for i, mat in enumerate(den_mat):
    mat = mat*0.5
    e_values , e_vectors = np.linalg.eigh(mat)
    print e_values
    print e_vectors
    rot_file.write('# site '+str(i)+'\n')
    rot_file.write(" ".join(str(x) for x in e_vectors[:,0])+'\n')
    rot_file.write(" ".join(str(x) for x in e_vectors[:,1])+'\n')
    rot_file.write('\n')
    #print np.dot(e_vectors[:,0:2].transpose(),np.dot(mat,e_vectors[:,0:2]))
rot_file.close()

# Generate and store PLOs
plo_converter.generate_and_output_as_text('plo.cfg', vasp_dir='./')

# run the archive creat routine
conv = VaspConverter('vasp')
conv.convert_dft_input()

sk = SumkDFTTools(hdf_file='vasp.h5')
