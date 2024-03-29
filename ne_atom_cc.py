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
Ne          0.0        0.0        0.0
'''

mol.charge = +0
mol.spin = +0
mol.max_memory = 1000 # MB


mol.basis = 'ccpvdz'


mol.build(cart=True)

mf = scf.RHF(mol).run()

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

#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CCD",method2="normal",diis=False)
#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CID",method2="normal",diis=False)
#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="directringCCD",method2="normal",diis=False)
#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="directringCCD+SOSEX",method2="normal",diis=False)
#E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="pCCD",method2="normal",diis=False, alpha=1.0, beta=0.0)
E_ccs, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="CCD",method2="singlet",diis=False)
E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=t2,method="CCD",method2="pairsinglet",diis=False)

e_tot_cc = E_cc + Escf 
e_tot_ccs = E_ccs + Escf 

print("         Energy")
print("ESCF                :%16.8f" %Escf)
print("CCD correlation     :%16.8f "%(E_cc))
print("CCD correlation     :%16.8f "%(E_ccs))
print("CCD full            :%16.8f "%(e_tot_ccs))
mycc = cc.CCSD(mf).run()
print(mycc.e_tot)
print(mycc.t1)
