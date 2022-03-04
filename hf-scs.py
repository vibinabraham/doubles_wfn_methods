import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from numpy import linalg as LA
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc, lo

np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)


###     PYSCF INPUT
r0 = 3.00

focc = 1
basis_set = '6-31g**'
basis_set = 'def2-tzvp'
basis_set = '6-31g'

r0 = 0.958 *  3
r0 = 2.4

molecule = '''
N
N   1   {}
'''.format(r0)


#r0 = 0.78
#molecule = '''
#H
#H   1   {}
#'''.format(r0)


#molecule = '''
#O
#H   1   {} 
#H   1   {}   2   104.5
#'''.format(r0,r0)




for ri in range(0,40):
    r0 = 0.529177*(1+ri*0.2)
    #r0 = 0.529177*(7)
    #r0 = 4.23
    basis_set = 'dzvp'
    molecule = '''
    H
    F   1   {}
    '''.format(r0)

    #PYSCF inputs
    mol = gto.Mole(atom=molecule,
        symmetry = True,basis = basis_set ,spin=0)
    mol.build()
    print("symmertry: ",mol.topgroup)

    cas_norb = mol.nao_nr() -focc
    cas_nel = mol.nelectron - 2* focc
    occ = cas_nel//2

    #SCF 
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.init_guess =  'atom'
    try:
        mf.run(max_cycle=300,dm0=dm)
    except:
        mf.run(max_cycle=300)
    dm = mf.make_rdm1()


    emp2 = mf.MP2().run(frozen=1)
    eccsd = mf.CCSD().run(frozen=1)

    print(mf.mo_coeff.shape)
    print(mf.e_tot)

    C = mf.mo_coeff[:,focc:]
    mo_energy = mf.mo_energy[focc:]
    occup = mf.get_occ(mf.mo_energy)
    occup = occup[focc:]
    #C = lo.PM(mol, mf.mo_coeff[:, focc:]).kernel(verbose=4)

    #molden.from_mo(mol, 'cas.molden', mf.mo_coeff)


    from pyscf import symm
    mo = symm.symmetrize_orb(mol, C)
    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    for i in range(len(osym)):
        print("%4d %8s %16.8f %8.4f"%(i+1,osym[i],mo_energy[i],occup[i]))


    ecore = mf.energy_nuc()
    h = C.T.dot(mf.get_hcore()).dot(C)
    g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((cas_norb),))

    E_cc2, t2  = run_ccd_method(mo_energy,h,g,cas_nel//2,t2= None ,method="DCD"   ,method2="normal",diis=False,damp_ratio = .95)
    print("DCD %16.8f"%(mf.e_tot+E_cc2))


    gap = mo_energy[occ] - mo_energy[occ-1]
    print("GAP:%8.4f"%gap)
    a = (np.sqrt(2*gap))**(1-gap)
    b = (np.sqrt(2*(1-gap)))**(gap)
    print(" ss:%8.4f"%a)
    print(" os:%8.4f"%b)
    Ec = run_mp2(mo_energy,h,g,cas_nel//2)
    print("      MP2:%16.8f"%(Ec+mf.e_tot))
    Ec2 = run_scs_mp2(mo_energy,h,g,cas_nel//2)
    Ec3 = run_scs_mp2(mo_energy,h,g,cas_nel//2,ss=b,os=a)

    print("r0:%6.2f   CCSD:%12.8f   DCD:%12.8f   MP2:%12.8f  SCS-MP2:%12.8f RM-MP2:%12.8f"%(r0,eccsd.e_tot,E_cc2+mf.e_tot,Ec+mf.e_tot,Ec2+mf.e_tot,Ec3+mf.e_tot))
