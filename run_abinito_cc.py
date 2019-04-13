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


mol = gto.Mole()
mol.atom = '''
N          0.0        0.0        0.0
N          0.0        0.0        1.0'''

mol.charge = +0
mol.spin = +0
mol.max_memory = 1000 # MB


mol.basis = 'sto-3g'

mol.build()

mf = scf.RHF(mol).run()


C = mf.mo_coeff

hcore = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
T = mol.intor('int1e_kin_sph')
V = mol.intor('int1e_nuc_sph') 
S = mol.intor('int1e_ovlp_sph')

g = mol.intor('int2e_sph')
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

E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CCD",method2="normal",diis=False)
E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CID",method2="normal",diis=False)
E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="directringCCD",method2="normal",diis=False)
E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="directringCCD+SOSEX",method2="normal",diis=False)
E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="pCCD",method2="normal",diis=False, alpha=1.0, beta=0.0)

e_tot_cc = E_cc + Escf

print("         Energy")
print("ESCF                :%16.8f" %Escf)
print("CCD correlation     :%16.8f "%(E_cc))
print("CCD full            :%16.8f "%(e_tot_cc))
