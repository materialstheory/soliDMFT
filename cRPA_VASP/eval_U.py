#!/usr/bin/python

import numpy as np
import collections

'''
python functions for reading Uijkl from a VASP cRPA run and the evaluating the matrix
elements for different basis sets.

Copyright (C) 2019, A. Hampel and C. Ederer from Materials Theory Group
at ETH Zurich
'''

def read_uijkl(path_to_uijkl,n_sites,n_orb):
    '''
    reads the VASP UIJKL files or the vijkl file if wanted

    Parameters
    ----------
    path_to_uijkl : string
        path to Uijkl like file
    n_sites: int
        number of different atoms (Wannier centers)
    n_orb : int
        number of orbitals per atom

    __Returns:__
    uijkl : numpy array
        uijkl Coulomb tensor

    '''
    dim = n_sites*n_orb
    uijkl = np.zeros((dim,dim,dim,dim))
    data =np.loadtxt(path_to_uijkl)

    for line in range(0,len(data[:,0])):
        i = int(data[line,0])-1
        j = int(data[line,1])-1
        k = int(data[line,2])-1
        l = int(data[line,3])-1
        uijkl[i,j,k,l] = data[line,4]

    return uijkl

def red_to_2ind(uijkl,n_sites,n_orb,out=False):
    '''
    reduces the 4index coulomb matrix to a 2index matrix and
    follows the procedure given in PRB96 seth,peil,georges:
    m = ii , m'=jj
    U_antipar = U_mm'^oo' = U_mm'mm' (Coulomb Int)
    U_par = U_mm'^oo = U_mm'mm' - U_mm'm'm (for intersite interaction)
    U_ijij (Hunds coupling)

    the indices in VASP are switched: U_ijkl ---VASP--> U_ikjl

    Parameters
    ----------
    uijkl : numpy array
        4d numpy array of Coulomb tensor
    n_sites: int
        number of different atoms (Wannier centers)
    n_orb : int
        number of orbitals per atom
    out : bool
        verbose mode

    __Returns:__
    Uij_anti : numpy array
        red 2 index matrix U_mm'mm'
    Uijij : numpy array
        red 2 index matrix U_ijij (Hunds coupling)
    Uijji : numpy array
        red 2 index matrix Uijji
    Uij_par : numpy array
        red 2 index matrix U_mm\'mm\' - U_mm\'m\'m

    '''
    dim = n_sites*n_orb

    # create 2 index matrix
    Uij_anti = np.zeros((dim, dim))
    Uij_par = np.zeros((dim, dim))
    Uijij = np.zeros((dim, dim))
    Uijji = np.zeros((dim, dim))

    for i in range(0,n_orb*n_sites):
        for j in range(0,n_orb*n_sites):
            # the indices in VASP are switched: U_ijkl ---VASP--> U_ikjl
            Uij_anti[i,j] = uijkl[i,i,j,j]
            Uijij[i,j] =  uijkl[i,j,i,j]
            Uijji[i,j] = uijkl[i,j,j,i]
            Uij_par[i,j] = uijkl[i,i,j,j]-uijkl[i,j,j,i]

    np.set_printoptions(precision=3,suppress=True)

    if out:
        print 'reduced U anti-parallel = U_mm\'\^oo\' = U_mm\'mm\' matrix : \n', Uij_anti
        print 'reduced Uijij : \n', Uijij
        print 'reduced Uijji : \n', Uijji
        print 'reduced U parallel = U_mm\'\^oo = U_mm\'mm\' - U_mm\'m\'m matrix : \n', Uij_par

    return Uij_anti,Uijij,Uijji,Uij_par


def calc_kan_params(uijkl,n_sites,n_orb,out=False):
    '''
    calculates the kanamori interaction parameters from a
    given Uijkl matrix. Follows the procedure given in
    PHYSICAL REVIEW B 86, 165105 (2012) Vaugier,Biermann
    formula 30,31,32
    Parameters
    ----------
    uijkl : numpy array
        4d numpy array of Coulomb tensor
    n_sites: int
        number of different atoms (Wannier centers)
    n_orb : int
        number of orbitals per atom
    out : bool
        verbose mode
    __Returns:__
    int_params : direct
        kanamori parameters
    '''

    int_params = collections.OrderedDict()
    dim = n_sites*n_orb
    Uij_anti,Uijij,Uijji,Uij_par = red_to_2ind(uijkl,n_sites,n_orb,out)

    # calculate intra-orbital U
    U = 0.0
    for i in range(0,n_orb):
        U += uijkl[i,i,i,i]
    U = U/(n_orb)
    int_params['U'] = U

    # calculate the U'
    Uprime = 0.0
    for i in range(0,n_orb):
        for j in range(0,n_orb):
            if i != j:
                Uprime +=  uijkl[i,i,j,j]
    Uprime = Uprime/ (n_orb*(n_orb-1))
    int_params['Uprime'] = Uprime

    # calculate J
    J = 0.0
    for i in range(0,n_orb):
        for j in range(0,n_orb):
            if i != j:
                J +=  uijkl[i,j,i,j]
    J = J/ (n_orb*(n_orb-1))
    int_params['J'] = J

    if out:
        print 'U= ', "{:.4f}".format(U)
        print 'U\'= ', "{:.4f}".format(Uprime)
        print 'J= ', "{:.4f}".format(J)

    return int_params

def calc_u_avg_fulld(uijkl,n_sites,n_orb,out=False):
    '''
    calculates the coulomb integrals from a
    given Uijkl matrix for full d shells. Follows the procedure given
    in Pavarini  - 2014 - arXiv - 1411 6906 - julich school U matrix
    page 8 or as done in
    PHYSICAL REVIEW B 86, 165105 (2012) Vaugier,Biermann
    formula 23, 25
    works atm only for full d shell (l=2)

    Returns F0=U, and J=(F2+F4)/2
    Parameters
    ----------
    uijkl : numpy array
        4d numpy array of Coulomb tensor
    n_sites: int
        number of different atoms (Wannier centers)
    n_orb : int
        number of orbitals per atom
    out : bool
        verbose mode
    __Returns:__
    int_params : direct
        Slater parameters
    '''

    int_params = collections.OrderedDict()
    dim = n_sites*n_orb
    Uij_anti,Uijij,Uijji,Uij_par = red_to_2ind(uijkl,n_sites,n_orb,out=out)
    # U_antipar = U_mm'^oo' = U_mm'mm' (Coulomb Int)
    # U_par = U_mm'^oo = U_mm'mm' - U_mm'm'm (for intersite interaction)
    # U_ijij (Hunds coupling)
    # here we assume cubic harmonics (real harmonics) as basis functions in the order
    # dz2 dxz dyz dx2-y2 dxy
    # triqs basis: basis ordered as (“xy”,”yz”,”z^2”,”xz”,”x^2-y^2”)

    # calculate J
    J_cubic = 0.0
    for i in range(0,n_orb):
        for j in range(0,n_orb):
            if i != j:
                J_cubic +=  Uijji[i,j]
    J_cubic = J_cubic/ (20)
    # 20 for 2l(2l+1)
    int_params['J_cubic'] = J_cubic

    # conversion from cubic to spherical:
    J = 7 * J_cubic / 5

    int_params['J'] = J

    # calculate intra-orbital U
    U_0 = 0.0
    for i in range(0,n_orb):
            U_0 += Uij_anti[i,i]
    U_0 = U_0 / n_orb
    int_params['U_0'] = U_0

    # now conversion from cubic to spherical
    U = U_0 - ( 8*J_cubic/5 )

    int_params['U'] = U

    if out:
        print 'cubic U_0= ', "{:.4f}".format(U_0)
        print 'cubic J_cubic= ', "{:.4f}".format(J_cubic)
        print 'spherical F0=U= ', "{:.4f}".format(U)
        print 'spherical J=(F2+f4)/14 = ', "{:.4f}".format(J)

    return int_params
