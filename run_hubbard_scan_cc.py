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
start_ratio =  1.00
stop_ratio  =  0.10
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

dcde = []
ccdt_e = []
ex_ccdt = []
dcdt = []
cc3_e = []
ccm3_e = []
exacpd14t = []
dcdut = []
beta = []
for i in range(0,n_steps+1):

    ratio =  start_ratio - i * step_size

    orb2 = (ratio/start_ratio) * orb
    h2 =  (ratio/start_ratio)  * h

    beta.append(ratio)

    print("Current ratio %16.8f" %ratio)
    E_cc, t2   = run_ccd_method(orb,h,g,n_elec//2,t2=t2,method="DCD",method2="normal",diis_start=40)
