import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from cid import *
from numpy import linalg as LA
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc, lo

np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)


###     PYSCF INPUT
r0 = 3.00

def get_h10chain_geom(r):
# {{{
    a1z = 9.*r/2
    a2z = 7.*r/2
    a3z = 5.*r/2
    a4z = 3.*r/2
    a5z = r/2.
    a6z = -r/2.
    a7z = -3.*r/2
    a8z = -5.*r/2
    a9z = -7.*r/2
    a10z = -9.*r/2

    g1 = '         '
    g2 = '     '


    geom =  'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a1z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a2z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a3z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a4z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a5z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a6z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a7z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a8z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a9z)  + '\n' + \
            'H' + g1 + str(0.00)  + g2 + str(0.00)  + g2 + str(a10z)

    return geom
# }}}

def get_h10ring_geom(r):
# {{{
    phi_o = 2 * np.pi / 10
    rad_denom = 2 * np.sin(np.pi/10)
    rad = r / rad_denom

    a1x = rad * np.cos(phi_o * 0)
    a2x = rad * np.cos(phi_o * 1)
    a3x = rad * np.cos(phi_o * 2)
    a4x = rad * np.cos(phi_o * 3)
    a5x = rad * np.cos(phi_o * 4)
    a6x = rad * np.cos(phi_o * 5)
    a7x = rad * np.cos(phi_o * 6)
    a8x = rad * np.cos(phi_o * 7)
    a9x = rad * np.cos(phi_o * 8)
    a10x = rad * np.cos(phi_o * 9)


    a1y = rad * np.sin(phi_o * 0)
    a2y = rad * np.sin(phi_o * 1)
    a3y = rad * np.sin(phi_o * 2)
    a4y = rad * np.sin(phi_o * 3)
    a5y = rad * np.sin(phi_o * 4)
    a6y = rad * np.sin(phi_o * 5)
    a7y = rad * np.sin(phi_o * 6)
    a8y = rad * np.sin(phi_o * 7)
    a9y = rad * np.sin(phi_o * 8)
    a10y = rad * np.sin(phi_o * 9)

    g1 = '         '
    g2 = '     '

    geom =  'H' + g1 + str(a1x)  + g2 + str(a1y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a2x)  + g2 + str(a2y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a3x)  + g2 + str(a3y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a4x)  + g2 + str(a4y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a5x)  + g2 + str(a5y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a6x)  + g2 + str(a6y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a7x)  + g2 + str(a7y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a8x)  + g2 + str(a8y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a9x)  + g2 + str(a9y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a10x) + g2 + str(a10y) + g2 + str(0.00)

    return geom
# }}}

def get_h10pyr_geom(r):
# {{{
    hpyr = np.sqrt(2.0/3) * r
    htri = np.cos(np.pi / 6.0) * r

    a1x =  0.0
    a2x = -0.5 * r
    a3x =  0.5 * r
    a4x = -r
    a5x =  0.0
    a6x =  r
    a7x =  0.0
    a8x = -0.5 * r
    a9x =  0.5 * r
    a10x = 0.0

    a1y =  (4.0/3) * htri
    a2y =  (1.0/3) * htri
    a3y =  (1.0/3) * htri
    a4y = -(2.0/3) * htri
    a5y = -(2.0/3) * htri
    a6y = -(2.0/3) * htri
    a7y =  (2.0/3) * htri
    a8y = -(1.0/3) * htri
    a9y = -(1.0/3) * htri
    a10y =  0.0

    a1z = -hpyr
    a2z = -hpyr
    a3z = -hpyr
    a4z = -hpyr
    a5z = -hpyr
    a6z = -hpyr
    a7z =  0.0
    a8z =  0.0
    a9z =  0.0
    a10z = hpyr

    g1 = '         '
    g2 = '     '


    geom =  'H' + g1 + str(a1x)  + g2 + str(a1y)  + g2 + str(a1z)  + '\n' + \
            'H' + g1 + str(a2x)  + g2 + str(a2y)  + g2 + str(a2z)  + '\n' + \
            'H' + g1 + str(a3x)  + g2 + str(a3y)  + g2 + str(a3z)  + '\n' + \
            'H' + g1 + str(a4x)  + g2 + str(a4y)  + g2 + str(a4z)  + '\n' + \
            'H' + g1 + str(a5x)  + g2 + str(a5y)  + g2 + str(a5z)  + '\n' + \
            'H' + g1 + str(a6x)  + g2 + str(a6y)  + g2 + str(a6z)  + '\n' + \
            'H' + g1 + str(a7x)  + g2 + str(a7y)  + g2 + str(a7z)  + '\n' + \
            'H' + g1 + str(a8x)  + g2 + str(a8y)  + g2 + str(a8z)  + '\n' + \
            'H' + g1 + str(a9x)  + g2 + str(a9y)  + g2 + str(a9z)  + '\n' + \
            'H' + g1 + str(a10x) + g2 + str(a10y) + g2 + str(a10z)

    return geom
# }}}

def get_h10sheet_geom(r):
# {{{
    htri = np.cos(np.pi / 6.0) * r

    a1x = -r
    a2x =  0.0
    a3x =  r
    a4x = -(3.0/2) * r
    a5x = -r/2
    a6x =  r/2
    a7x = (3.0/2) * r
    a8x = -r
    a9x =  0.0
    a10x =  r

    a1y = htri
    a2y = htri
    a3y = htri
    a4y = 0.0
    a5y = 0.0
    a6y = 0.0
    a7y = 0.0
    a8y = -htri
    a9y = -htri
    a10y = -htri

    g1 = '         '
    g2 = '     '

    geom =  'H' + g1 + str(a1x)  + g2 + str(a1y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a2x)  + g2 + str(a2y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a3x)  + g2 + str(a3y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a4x)  + g2 + str(a4y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a5x)  + g2 + str(a5y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a6x)  + g2 + str(a6y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a7x)  + g2 + str(a7y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a8x)  + g2 + str(a8y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a9x)  + g2 + str(a9y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a10x) + g2 + str(a10y) + g2 + str(0.00)

    return geom
# }}}


#molecule =  get_h10chain_geom(r0)
#molecule =  get_h10ring_geom(r0)
#molecule =  get_h10sheet_geom(r0)
molecule =  get_h10pyr_geom(r0)


focc = 0
basis_set = 'sto-6g'

cas_norb = 10
cas_nel = 10



for ind in range(0,41):
    r0 = 0.8 + 0.05 * ind
    molecule =  get_h10sheet_geom(r0)
    #PYSCF inputs
    mol = gto.Mole(atom=molecule,
	symmetry = False,basis = basis_set ,spin=0)
    mol.build()
    print("symmertry: ",mol.topgroup)

    #SCF 
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.run(max_cycle=300)

    print(mf.mo_coeff.shape)

    C = mf.mo_coeff[:,focc:]
    mo_energy = mf.mo_energy[focc:]
    #C = lo.PM(mol, mf.mo_coeff[:, focc:]).kernel(verbose=4)

    #molden.from_mo(mol, 'cas.molden', mf.mo_coeff)


    #from pyscf import symm
    #mo = symm.symmetrize_orb(mol, C)
    #osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    ##symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    #for i in range(len(osym)):
    #    print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))


    ecore = mf.energy_nuc()
    h = C.T.dot(mf.get_hcore()).dot(C)
    g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((cas_norb),))

    try:
        E_cc, t2  = run_ccd_method(mo_energy,h,g,cas_nel//2,t2= t2 ,method="DCD"   ,method2="normal",diis=False,damp_ratio = .9)
    except:
        E_cc, t2  = run_ccd_method(mo_energy,h,g,cas_nel//2,t2= None ,method="DCD"   ,method2="normal",diis=False,damp_ratio = .9)

    print("DCD %16.8f"%(mf.e_tot+E_cc))
    E_cc, t2  = run_ccd_method(mo_energy,h,g,cas_nel//2,t2= t2 ,method="CCD"   ,method2="pairsinglet",diis=False,damp_ratio = .9)
    print("pairCCD %16.8f"%(mf.e_tot+E_cc))
    print(t2.shape)
    print(E_cc)
    print(E_cc+mf.e_tot)
    print(mf.e_tot)

    if 1:
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        cisolver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), shift=.01)
        ecas, vcas = cisolver.kernel(h, g, h.shape[0], nelec=cas_nel, ecore=ecore,nroots =2,verbose=100)
        #print("FCI %12.8f"%ecas)

        #print('E = %.12f  2S+1 = %.7f' %
        #          (ecas, cisolver.spin_square(vcas, h.shape[0], (cas_nel//2,cas_nel//2))[1]))

        print('E = energy for r',r0)
        for i in range(ecas.shape[0]):
            print('State:%d E = %.12f  2S+1 = %.7f' %(i,ecas[i], cisolver.spin_square(vcas[i], h.shape[0], (5,5))[1]))
    exit()


