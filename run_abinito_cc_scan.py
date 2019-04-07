import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc
from numpy import linalg as LA
np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)


r = 0.7
radius = []
all_scf = []
all_mp2 = []
all_ccd = []
all_ccd0 = []
all_ccd1 = []
all_dcd = []
all_acd0 = []
all_acd1 = []

for Iter in range(0,3):
    r += 0.1
    mol = gto.Mole()
    mol.atom = '''
    N          0.0        0.0        0.0
    N          0.0        0.0        '''+str(r)

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


    print("HOMO-LUMO gap %16.10f "%(E_occ[-1] - E_vir[0]))

    F = mf.get_fock()
    energyhf = mf.e_tot


    he  =  np.einsum('ii',h[:nel,:nel]) 
    Je  =  np.einsum('ppqq',g[:nel,:nel,:nel,:nel])
    Ke  =  np.einsum('pqqp',g[:nel,:nel,:nel,:nel])

    Escf = 2 * he +  2 * Je - Ke

    Emp2 = run_mp2(orb,h,g,nel)
    if Iter ==0:
        Edc, t2d  = run_ccd_method(orb,h,g,nel,t2=None,method="DCD",method2="normal",diis=False)
        Ecc, t2   = run_ccd_method(orb,h,g,nel,t2d,method="CCD",method2="normal",diis_start=30)
    else:
        Edc, t2d  = run_ccd_method(orb,h,g,nel,t2d,method="DCD",method2="normal",diis=False)
        Ecc, t2   = run_ccd_method(orb,h,g,nel,t2d,method="CCD",method2="normal",diis_start=30)

    mp2 = mp.MP2(mf)
    mp2.kernel()
    energymp2 = mp2.e_tot
    assert(np.isclose(energymp2, Escf + Emp2 + E_nu))
    assert(np.isclose(energyhf, Escf + E_nu))

    print("radius%16.8f"%(r))
    e_tot_scf = Escf+E_nu
    e_tot_mp2 = Escf+E_nu+Emp2
    e_tot_dcd = Edc+Escf+E_nu
    e_tot_ccd = Ecc+Escf+E_nu

    print(" Eccd        %16.8f "%(e_tot_ccd))
    print(" Edcd        %16.8f "%(e_tot_dcd))

    radius.append(r)
    all_scf.append(e_tot_scf )
    all_mp2.append(e_tot_mp2 )
    all_dcd.append(e_tot_dcd )
    all_ccd.append(e_tot_ccd )

for i in range(len(radius)):
    print("")
    print("radius%16.8f"%(radius[i]))
    print(" Escf %16.8f "%(all_scf[i]))
    print(" Emp2 %16.8f "%(all_mp2[i]))
    print(" Eccd %16.8f "%(all_ccd[i]))
    print(" Edcd %16.8f "%(all_dcd[i]))
