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
n_site = 6
U = 13
beta = 1.0

#Active Space
n_orb  = n_site
n_elec = 6
nel = n_elec//2

#list of kappa to be tested
kappa = [0,0.3,0.4,0.7]


for U in range(1,30):
    U = float(U)
    h_local,g_local = get_hubbard_params(n_site,beta,U,pbc=True)


    print("\nHubbard Hamiltonian\n")

    print("Number of orbitals     : {}".format(n_orb))
    print("Coulomb Repulsion (U)  : {}".format(U))
    print("Tight binding (t)      : \n{}".format(h_local))
    print()

    Escf,orb,h,g,C = run_hubbard_scf(h_local,g_local,n_elec//2,h_local)
    g = np.einsum("pqrs,pl->lqrs",g_local,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)

    h = C.T @ h_local @ C

    # run mp2
    run_mp2(orb,h,g,nel)

    # run the xbw
    run_bw(orb,h,g,nel)

    # run the kappa mp2
    for ki in kappa:
        run_kmp2(orb,h,g,nel,kappa=ki)

    if 0:
        E_cc1 , t2   = run_ccd_method(orb,h,g,nel,t2= None  ,method="ACPD45"     ,method2="normal",diis=False,damp_ratio=0.9)
        E_cc3 , t2   = run_ccd_method(orb,h,g,nel,t2= t2   ,method="DCD"        ,method2="normal",diis=False,damp_ratio=0.98)
        print("Energy")
        print("DCD          %16.8f" %(Escf+E_cc3 ))

        cisolver = fci.direct_spin1.FCI()
        cisolver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), shift=.01)
        ecas, vcas = cisolver.kernel(h, g, h.shape[0], nelec=n_elec, ecore=0,nroots =1,verbose=0)
        print("FCI %12.8f"%ecas)

