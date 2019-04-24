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
U = 5

#Active Space
n_orb   = 6
n_elec  = 6
ms      = 0     #only takes positive numbers (coded such that \alpha string more that \beta always)


k = n_elec // 2 + ms
n_a = max(k, n_elec - k)
n_b = min(k, n_elec - k)



#loop inputs
start_ratio =  0.80
stop_ratio  =  0.60
step_size   =  0.10

######### Integrals (local) hubbard
t = np.zeros((n_orb,n_orb))
for i in range(0,n_orb-1):
    t[i,i+1] = 1
    t[i+1,i] = 1
t[0,n_orb-1] = 1
t[n_orb-1,0] = 1

h_local = -start_ratio  * t 
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

E_cc, t2   = run_ccd_method(orb,h,g,n_elec//2,t2=None,method="DCD",method2="normal",diis_start=40)

n_steps = int((start_ratio - stop_ratio)/step_size)
assert(n_steps > 0)
print("\nNumber of Calculations " ,n_steps)

nocc = n_elec//2
nvir = h_local.shape[0] - n_elec//2
c2 = np.zeros((nocc,nocc,nvir,nvir))

dcd_e = []
ccd_e = []
ccd0_e = []
ccd1_e = []
cid_e = []
cid0_e = []
cid1_e = []
beta = []
for i in range(0,n_steps+1):

    ratio =  start_ratio - i * step_size

    orb2 = (ratio/start_ratio) * orb
    h2 =  (ratio/start_ratio)  * h

    beta.append(ratio)

    print("Current ratio %16.8f" %ratio)
    dcd_ee, t2d   = run_ccd_method(orb2,h2,g,n_elec//2,t2=t2,method="DCD",method2="normal",diis_start=40)
    ccd_ee, t2   = run_ccd_method(orb2,h2,g,n_elec//2,t2d,method="CCD",method2="normal",diis_start=50)
    ccd0_ee, t2s = run_ccd_method(orb2,h2,g,n_elec//2,t2d,method="CCD",method2="singlet",diis=False)
    ccd1_ee, t2t = run_ccd_method(orb2,h2,g,n_elec//2,t2=None,method="CCD",method2="triplet", diis=False)
    cid_ee, c2   = run_ccd_method(orb2,h2,g,n_elec//2,c2,method="CID",method2="normal",diis=False, damp_ratio=0.9)
    cid0_ee, c2_s   = run_ccd_method(orb2,h2,g,n_elec//2,c2,method="CID",method2="singlet",diis=False)
    cid1_ee, c2_t   = run_ccd_method(orb2,h2,g,n_elec//2,c2,method="CID",method2="triplet",diis=False)


    dcd_e.append(dcd_ee)
    ccd_e.append(ccd_ee)
    ccd0_e.append(ccd0_ee)
    ccd1_e.append(ccd1_ee)
    cid_e.append(cid_ee)
    cid0_e.append(cid0_ee)
    cid1_e.append(cid1_ee)


for i in range(len(beta)): 
    print("Current beta %16.8f" %beta[i])
    print("DCD  %16.8f" %(dcd_e[i]/6))
    print("CCD  %16.8f" %(ccd_e[i]/6))
    print("CCD0 %16.8f" %(ccd0_e[i]/6))
    print("CCD1 %16.8f" %(ccd1_e[i]/6))
    print("CID  %16.8f" %(cid_e[i]/6))
    print("CID0 %16.8f" %(cid0_e[i]/6))
    print("CID1 %16.8f" %(cid1_e[i]/6))
