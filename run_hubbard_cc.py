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
n_cene = 2
U = 5
beta = 8


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

Escf,orb,h,g,C = run_hubbard_scf(5 * h_local,g_local,n_elec//2,t)
E_cc, t2,cc   = run_ccd_method(orb,h,g,nel,t2=None,method="ACPD45",method2="normal",diis=False,damp_ratio=0.9)
E_cc, c2,cc   = run_ccd_method(orb,h,g,nel,t2=t2  ,method="CID"   ,method2="normal",diis=False,damp_ratio=0.9)

Escf,orb,h,g,C = run_hubbard_scf(h_local,g_local,n_elec//2,t)

E_cc1 , t2  ,conv1   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="ACPD45"     ,method2="normal",diis=False,damp_ratio=0.95)
E_cc2 , t2_ ,conv2   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="ACPD14"     ,method2="normal",diis=False,damp_ratio=0.9)
E_cc3 , t2  ,conv3   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="DCD"        ,method2="normal",diis=False,damp_ratio=0.98)
E_cc4 , t2_ ,conv4   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="CCD"        ,method2="normal",diis=False,damp_ratio=0.99)
E_cc5 , t2_ ,conv5   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="LCC"        ,method2="normal",diis=False,damp_ratio=0.99)
E_cc6 , t2_ ,conv6   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="pCCD"       ,method2="normal",diis=False, alpha= 1.0, beta=0.0,damp_ratio = 0.95)
E_cc7 , t2_ ,conv7   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="pCCD"       ,method2="normal",diis=False, alpha=-1.0, beta=0.01,damp_ratio = 0.95)
E_cc8 , t2_ ,conv8   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="pCCD"       ,method2="normal",diis=False, alpha=-1.5, beta=0.01,damp_ratio = 0.95)
E_cc9 , t2_ ,conv9   = run_ccd_method(orb,h,g,nel,t2= None ,method="pCCD"       ,method2="normal",diis=False, alpha=-2.0, beta=0.01,damp_ratio = 0.95)
E_cc10, t2_ ,conv10  = run_ccd_method(orb,h,g,nel,t2= c2   ,method="CID"        ,method2="normal",diis=False,damp_ratio=0.99)
E_cc11, t2_ ,conv11  = run_ccd_method(orb,h,g,nel,t2= t2   ,method="CCD"        ,method2="singlet",diis=False,damp_ratio=0.98)
E_cc12, t2_ ,conv12  = run_ccd_method(orb,h,g,nel,t2= t2   ,method="CCD"        ,method2="triplet",diis=False,damp_ratio=0.98)
E_cc13, t2_ ,conv13  = run_ccd_method(orb,h,g,nel,t2= t2   ,method="CID"        ,method2="singlet",diis=False,damp_ratio=0.98)
E_cc14, t2_ ,conv14  = run_ccd_method(orb,h,g,nel,t2= t2   ,method="CID"        ,method2="triplet",diis=False,damp_ratio=0.98)
E_cc15, t2_ ,conv15  = run_ccd_method(orb,h,g,nel,t2= t2   ,method="directringCCD",method2="normal",diis=False,damp_ratio=0.98)
E_cc16, t2_ ,conv16  = run_ccd_method(orb,h,g,nel,t2= t2   ,method="directringCCD+SOSEX",method2="normal",diis=False,damp_ratio=0.98)


print("Energy/site")
print("ACPD45       %16.10f  %r" %(E_cc1 /n_orb,conv1 ))
print("ACPD14       %16.10f  %r" %(E_cc2 /n_orb,conv2 ))
print("DCD          %16.10f  %r" %(E_cc3 /n_orb,conv3 ))
print("CCD          %16.10f  %r" %(E_cc4 /n_orb,conv4 ))
print("LCC          %16.10f  %r" %(E_cc5 /n_orb,conv5 ))
print("pCCD         %16.10f  %r" %(E_cc6 /n_orb,conv6 ))
print("pCCD         %16.10f  %r" %(E_cc7 /n_orb,conv7 ))
print("pCCD         %16.10f  %r" %(E_cc8 /n_orb,conv8 ))
print("pCCD         %16.10f  %r" %(E_cc9 /n_orb,conv9 ))
print("CID          %16.10f  %r" %(E_cc10/n_orb,conv10))
print("CCDs         %16.10f  %r" %(E_cc11/n_orb,conv11))
print("CCDt         %16.10f  %r" %(E_cc12/n_orb,conv12))
print("CIDs         %16.10f  %r" %(E_cc13/n_orb,conv13))
print("CIDt         %16.10f  %r" %(E_cc14/n_orb,conv14))
print("drCC         %16.10f  %r" %(E_cc15/n_orb,conv15))
print("drCCSOSex    %16.10f  %r" %(E_cc16/n_orb,conv16))
