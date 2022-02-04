# Benchmarking each method in multiple codes
# H10 1D Linear chain at r = 0.9 A
# STO-6G basis

# SCF 
# ACES2 - -5.276404037486 H
# pyscf - -5.27640419     H

# CCD
# ACES2 - -0.138521220751 H  
# pyscf - -0.138521215928 H
# our code - -0.138521205091

# LCCD
# ACES2 - -0.146258299056 H
# our code - -0.146258281462 H

# CID
# ACES2 - -0.129772215759
# our code - -0.12977220


# DCD
# ACES2 -    -0.138907096026 H
# our code - -0.138907083009 H

# parametrized CCD

# 2CC -pCCD(1,0)
# ACES2 - -0.139531293910 H
# our code - -0.139531270404 H


# pCCD(-1,1)
# ACES2 - -0.141424745217 H
# our code - -0.14142472768 H


# singlet CCD (CCD0)
# pyscf - -0.11701133938059298 H
# our code - -0.117011446410 H


# pair CCD
# pyscf - -0.03197149323902028 H 
# our code - -0.031971487833 H

import unittest

import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from numpy import linalg as LA

np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)





mol = gto.M(
        atom = '''
        H         0.0     0.0     4.05
        H         0.0     0.0     3.15
        H         0.0     0.0     2.25
        H         0.0     0.0     1.35
        H         0.0     0.0     0.45
        H         0.0     0.0     -0.45
        H         0.0     0.0     -1.35
        H         0.0     0.0     -2.25
        H         0.0     0.0     -3.15
        H         0.0     0.0     -4.05
        ''',
        verbose= 2,
        #basis= 'adzp',
        unit='Angstroms',
        #cart=True,
        symmetry = True,
        basis= 'sto6g',
    )



mf = scf.HF(mol).run()
c = mol.cart2sph_coeff()
print(c.shape)


C = mf.mo_coeff
print(C.shape)

hcore = mol.intor('int1e_nuc_cart') + mol.intor('int1e_kin_cart')
T = mol.intor('int1e_kin_cart')
V = mol.intor('int1e_nuc_cart')
S = mol.intor('int1e_ovlp_cart')



s0 = mol.intor('int1e_ovlp_sph')
s1 = c.T.dot(mol.intor('int1e_ovlp_cart')).dot(c)
print(abs(s1-s0).sum())

g = mol.intor('int2e_cart')
print(g.shape)
g = np.einsum("pqrs,pl->lqrs",g,C)
g = np.einsum("lqrs,qm->lmrs",g,C)
g = np.einsum("lmrs,rn->lmns",g,C)
g = np.einsum("lmns,so->lmno",g,C)

h = C.T @ hcore @ C


E_nu = gto.Mole.energy_nuc(mol)
nel = mol.nelectron//2
n_b , n_a = mol.nelec
n_orb = mol.nao_nr()
tot = n_orb


"""Orbital energies"""
#orb = np.asarray(wfn.epsilon_a())
orb = mf.mo_energy
E_occ = orb[:nel]
E_vir = orb[nel:]

F = mf.get_fock()
energyhf = mf.e_tot

he  =  np.einsum('ii',h[:nel,:nel])
Je  =  np.einsum('ppqq',g[:nel,:nel,:nel,:nel])
Ke  =  np.einsum('pqqp',g[:nel,:nel,:nel,:nel])

Escf = 2 * he +  2 * Je - Ke
Escf = Escf + E_nu


E_CCD, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CCD",method2="normal",diis=False)
E_LCCD, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="LCCD",method2="normal",diis=False)
E_CID, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CID",method2="normal",diis=False)
#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="directringCCD",method2="normal",diis=False)
#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="directringCCD+SOSEX",method2="normal",diis=False)
E_par_ccd, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="pCCD",method2="normal",diis=False, alpha=-1.0, beta=1.0)
E_par_2cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="pCCD",method2="normal",diis=False, alpha=1.0, beta=0.0)
E_DCD, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="DCD",method2="normal",diis=False)
E_singletCCD, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CCD",method2="singlet",diis=False)
E_pairCCD, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CCD",method2="pairsinglet",diis=False)


e_tot_CCD = E_CCD + Escf
e_tot_DCD = E_DCD + Escf
e_tot_singletccd = E_singletCCD + Escf
e_tot_pairccd = E_pairCCD + Escf



print('At r = 0.9 A')
print("         Energy")
print("ESCF                :%16.8f" %Escf)
print("ESCF (pyscf)               :%16.8f" %energyhf)
print("Singlet CCD correlation     :%16.8f "%(E_singletCCD))
print("pair CCD correlation     :%16.8f "%(E_pairCCD))
print("DCD correlation     :%16.8f "%(E_DCD))
print("CCD correlation     :%16.8f "%(E_CCD))
#print("CCD correlation     :%16.8f "%(E_ccs))
#print("CCD full            :%16.8f "%(e_tot_ccs))
#mycc = cc.CCSD(mf).run()
#print(mycc.e_tot)




class KnownValues(unittest.TestCase):
    def test_ccd(self):
        self.assertAlmostEqual(E_CCD, -0.138521205091, 6)

    def test_dcd(self):
        self.assertAlmostEqual(E_DCD, -0.138907083009, 6)

    def test_paramterized_ccd(self):
        self.assertAlmostEqual(E_par_ccd, -0.14142472768, 6)
        self.assertAlmostEqual(E_par_2cc, -0.139531270404, 6)

    def test_lccd(self):
        self.assertAlmostEqual(E_LCCD, -0.146258281462, 6)

    def test_cid(self):
        self.assertAlmostEqual(E_CID, -0.12977220, 6)

    def test_singletccd(self):
        self.assertAlmostEqual(E_singletCCD,-0.117011446410, 6)

    def test_pairccd(self):
        self.assertAlmostEqual(E_pairCCD,-0.031971487833, 6)


if __name__ == "__main__":
    print("Full Tests for CCD variants")
    unittest.main()

