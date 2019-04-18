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
n_cene = 1
U = 5
beta = 2.0


n_site,t,h_local,g_local = get_hubbard_params_ncene(n_cene,beta,U)

#Active Space
n_orb  = n_site
n_elec = n_site
nel = n_elec//2



print("\nHubbard Hamiltonian\n")

print("Number of orbitals     : {}".format(n_orb))
print("Coulomb Repulsion (U)  : {}".format(U))
print("Tight binding (t)      : \n{}".format(h_local))
print()


Escf,orb,h,g,C = run_hubbard_scf(h_local,g_local,n_elec//2,t)

E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="ACPD45",method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="ACPD14",method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="DCD"   ,method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CCD"   ,method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="LCC"   ,method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=1.0, beta=0.0)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=-1.0, beta=1.0,damp_ratio = 0.7)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=-1.5, beta=1.0,damp_ratio = 0.7)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=-2.0, beta=1.0,damp_ratio = 0.7)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CID"   ,method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CCD"   ,method2="singlet",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CCD"   ,method2="triplet",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="directringCCD",method2="normal",diis=False)
E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="directringCCD+SOSEX",method2="normal",diis=False)
