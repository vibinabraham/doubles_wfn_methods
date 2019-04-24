import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from numpy import linalg as LA
np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)


##Parameters
n_cene = 2
U = 5

#loop inputs (beta)
start_ratio =  1.00
stop_ratio  =  0.60
step_size   =  0.20



n_site,t,h_local,g_local = get_hubbard_params_ncene(n_cene,start_ratio,U)

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

E_cc, t2   = run_ccd_method(orb,h,g,n_elec//2,t2=None,method="DCD",method2="normal",diis_start=40)

n_steps = int((start_ratio - stop_ratio)/step_size)
assert(n_steps > 0)
print("\nNumber of Calculations " ,n_steps)

dcde = []
ccdt_e = []
ex_ccdt = []
dcdt = []
cc3_e = []
ccm3_e = []
exacpd14t = []
dcdut = []
beta_v = []
for i in range(0,n_steps+1):

    ratio =  start_ratio - i * step_size

    orb2 = (ratio/start_ratio) * orb
    h2 =  (ratio/start_ratio)  * h

    beta_v.append(ratio)

    print("Current ratio %16.8f" %ratio)
    E_cc, t2   = run_ccd_method(orb,h,g,nel,t2=None,method="ACPD45",method2="normal",diis=True)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="ACPD14",method2="normal",diis=True)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="DCD"   ,method2="normal",diis=True)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CCD"   ,method2="normal",diis=False)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="LCC"   ,method2="normal",diis=True)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=1.0, beta=0.0)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=True, alpha=-1.0, beta=1.0,damp_ratio = 0.9)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=-1.5, beta=1.0,damp_ratio = 0.9)
    #E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="pCCD"  ,method2="normal",diis=False, alpha=-2.0, beta=1.0,damp_ratio = 0.9)
    E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CID"   ,method2="normal",diis=False)
    E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CCD"   ,method2="singlet",diis=False)
    E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="CCD"   ,method2="triplet",diis=False)
    E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="directringCCD",method2="normal",diis=False)
    E_cc, t2_  = run_ccd_method(orb,h,g,nel,t2= t2 ,method="directringCCD+SOSEX",method2="normal",diis=False)
