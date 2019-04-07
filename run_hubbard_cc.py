import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from numpy import linalg as LA
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc

np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)


##Parameters
U = 2
t_ratio = 1.0

#Active Space
n_orb   = 6
n_elec  = 6
ms      = 0     #only takes positive numbers (coded such that \alpha string more that \beta always)


k = n_elec // 2 + ms
n_a = max(k, n_elec - k)
n_b = min(k, n_elec - k)



######### Integrals (local) hubbard
t = np.zeros((n_orb,n_orb))
for i in range(0,n_orb-1):
    t[i,i+1] = 1
    t[i+1,i] = 1
t[0,n_orb-1] = 1
t[n_orb-1,0] = 1

h_local = -t_ratio  * t 
g_local = np.zeros((n_orb,n_orb,n_orb,n_orb))
for i in range(0,n_orb):
    g_local[i,i,i,i] = U



print("\nHubbard Hamiltonian\n")

print("Number of orbitals     : {}".format(n_orb))
print("Number of alpha string : {}".format(n_a))
print("Number of beta string  : {}".format(n_b))
print("Coulomb Repulsion (U)  : {}".format(U))
print("Tight binding (t)      : \n{}".format(h_local))
print()


Escf,orb,h,g,C = run_hubbard_scf(h_local,g_local,n_elec//2,t)

E_dc, t2d   = run_ccd_method(orb,h,g,n_elec//2,t2=None,method="DCD",method2="normal",diis=False)
E_cc, t2   = run_ccd_method(orb,h,g,n_elec//2,t2d,method="CCD",method2="normal",diis_start=50)
E_ccs, t2s = run_ccd_method(orb,h,g,n_elec//2,t2=None,method="CCD",method2="singlet",diis=False)
E_cct, t2t = run_ccd_method(orb,h,g,n_elec//2,t2=None,method="CCD",method2="triplet")
t2 = t2.ravel()
t2s = t2s.ravel()
t2t = t2t.ravel()
t2d = t2d.ravel()

e_tot_cc = E_cc + Escf
e_tot_ccs = E_ccs + Escf
e_tot_cct = E_cct + Escf
e_tot_dc  = E_dc  + Escf
#help(run_ccd_method)
for i in range(t2.shape[0]):
    print("%16.8f %16.8f %16.8f %16.8f"%(t2[i],t2s[i],t2t[i],t2d[i]))
cisolver = fci.direct_spin1.FCI()
efci, ci = cisolver.kernel(h, g, h.shape[1], (n_a,n_b), ecore=0)

print("         Energy")
print("ESCF %16.8f" %Escf)
print(" FCI                  :%16.10f" %efci)
print("%16.8f %16.8f %16.8f %16.8f"%(E_cc,E_ccs,E_cct,E_dc))
print("         CCD              CCD0             CCD1            DCD")
print("%16.8f %16.8f %16.8f %16.8f"%(e_tot_cc,e_tot_ccs,e_tot_cct,e_tot_dc))
print(ci.shape)
print(ci)

