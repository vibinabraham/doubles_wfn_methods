import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from numpy import linalg as LA
np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc, lo
import scipy.linalg as la


molecule= '''
O
H 1 1.1
H 1 1.1 2 104
'''

basis_set = 'ccpvdz'

for ri in range(0,13):
    r0 = 2.8 + 0.1*ri 
    molecule = '''
    H
    H   1   {}
    '''.format(r0)


    mol = gto.Mole(atom=molecule,
        symmetry = False,basis = basis_set ,spin=0)
    mol.build()

    nao = mol.nao_nr()
    nocc = mol.nelectron
    focc = 0

    #SCF 
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.run(max_cycle=300)

    #print(mf.mo_coeff.shape)

    C = mf.mo_coeff[:,focc:]
    orb = mf.mo_energy[focc:]

    ecore = mf.energy_nuc()
    # Compute overlap integrals
    ovlp = mol.intor('cint1e_ovlp_sph')
    # Compute one-electron kinetic integrals
    T = mol.intor('cint1e_kin_sph')
    # Compute one-electron potential integrals
    V = mol.intor('cint1e_nuc_sph')
    # Compute two-electron repulsion integrals (Chemists’ notation)
    gao = mol.intor('cint2e_sph').reshape((nao,)*4)
    hao = T + V



    # Transform to spin orbital space
    eps,hao,gao,C = spatial_2_spin(mf.mo_energy,hao,gao,mf.mo_coeff)

    # Transform gao and hao into MO basis
    hmo,gmo = ao_to_mo(hao, gao, C)

    # Optimize orbitals
    emp2 = oomp2(eps,hmo,gmo,C,nocc,ecore)
    print("ooMP2%16.8f"%(emp2))

