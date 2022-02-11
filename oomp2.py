
import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from code import *
from numpy import linalg as LA
np.set_printoptions(precision=5, linewidth=200, suppress=True)
numpy_memory = 2 # numpy memory limit (2GB)
from pyscf import gto, scf, mcscf, fci, ao2mo, mp, cc, lo
import scipy.linalg as la


molecule= '''
O
H 1 1.1
H 1 1.1 2 104
'''

molecule= '''
H 0 0 0
H 0 0 2
H 0 2 0
H 0 2 2
'''

basis_set = '6-31g'

mol = gto.Mole(atom=molecule,
    symmetry = False,basis = basis_set ,spin=0)
mol.build()

nao = mol.nao_nr()
nel,nel = mol.nelec
focc = 0

#SCF 
mf = scf.RHF(mol)
mf.verbose = 0
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-10
mf.run(max_cycle=300)

#print(mf.mo_coeff.shape)

C = mf.mo_coeff[:,focc:]
orb = mf.mo_energy[focc:]

E_nuc = mf.energy_nuc()
# Compute overlap integrals
ovlp = mol.intor('cint1e_ovlp_sph')
# Compute one-electron kinetic integrals
T = mol.intor('cint1e_kin_sph')
# Compute one-electron potential integrals
V = mol.intor('cint1e_nuc_sph')
# Compute two-electron repulsion integrals (Chemists’ notation)
gao = mol.intor('cint2e_sph').reshape((nao,)*4)
h = T + V


# ==> Set default program options <==
# Maximum OMP2 iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-8



nbf = h.shape[0]          # Number of basis functions
nalpha = nel  # Number of alpha electrons
nbeta =  nel    # Number of beta electrons
nocc = nalpha + nbeta      # Total number of electrons
nso = 2 * nbf              # Total number of spin orbitals
nvirt = nso - nocc         # Number of virtual orbitals



#code block7{{{
def spin_block_tei(I):
    '''
    Spin blocks 2-electron integrals
    Using np.kron, we project I and I tranpose into the space of the 2x2 ide
    The result is our 2-electron integral tensor in spin orbital notation
    '''
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)
 
I = np.asarray(gao)
I_spinblock = spin_block_tei(I)
 
# Convert chemist's notation to physicist's notation, and antisymmetrize
# (pq | rs) ---> <pr | qs>
# <pr||qs> = <pr | qs> - <pr | sq>
gao = I_spinblock.transpose(0, 2, 1, 3) - I_spinblock.transpose(0, 2, 3, 1)
 # }}}


#code block8{{{
# ==> core Hamiltoniam <==

h = np.asarray(h)

# Using np.kron, we project h into the space of the 2x2 identity
# The result is the core Hamiltonian in the spin orbital formulation
hao = np.kron(np.eye(2), h)
# }}}

#code block 9{{{
# Get orbital energies, cast into NumPy array, and extend eigenvalues
eps_a = np.asarray(mf.mo_energy)
eps_b = np.asarray(mf.mo_energy)
eps = np.append(eps_a, eps_b)

# Get coefficients, block, and sort
Ca = np.asarray(C)
Cb = np.asarray(C)
C = np.block([
             [      Ca,         np.zeros_like(Cb)],
             [np.zeros_like(Ca),          Cb     ]])

# Sort the columns of C according to the order of orbital energies
C = C[:, eps.argsort()]
# }}}
 
# code block 10 {{{
# ==> AO to MO transformation functions <==


def ao_to_mo(hao, C):
    '''
    Transform hao, which is the core Hamiltonian in the spin orbital basis,
    into the MO basis using MO coefficients
    '''
    
    return np.einsum('pQ, pP -> PQ', 
           np.einsum('pq, qQ -> pQ', hao, C, optimize=True), C, optimize=True)


def ao_to_mo_tei(gao, C):
    '''
    Transform gao, which is the spin-blocked 4d array of physicist's notation,
    antisymmetric two-electron integrals, into the MO basis using MO coefficients
    '''
    
    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS', 
           np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

# Transform gao and hao into MO basis
hmo = ao_to_mo(hao, C)
gmo = ao_to_mo_tei(gao, C)
# }}}

# {{{ code block 11
# Make slices
o = slice(None, nocc)
v = slice(nocc, None)
x = np.newaxis
# }}}
# Intialize t amplitude and energy 
t_amp = np.zeros((nvirt, nvirt, nocc, nocc)) 
E_OMP2_old = 0.0 

# Initialize the correlation one particle density matrix
opdm_corr = np.zeros((nso, nso))

# Build the reference one particle density matrix
opdm_ref = np.zeros((nso, nso))
opdm_ref[o, o] = np.identity(nocc)

# Initialize two particle density matrix
tpdm_corr = np.zeros((nso, nso, nso, nso))

# Initialize the rotation matrix parameter 
X = np.zeros((nso, nso))

for iteration in range(MAXITER):

    # Build the Fock matrix
    f = hmo + np.einsum('piqi -> pq', gmo[:, o, :, o], optimize=True)

    # Build off-diagonal Fock Matrix and orbital energies
    fprime = f.copy()
    np.fill_diagonal(fprime, 0)
    eps = f.diagonal()

    # Update t amplitudes
    t1 = gmo[v, v, o, o]
    t2 = np.einsum('ac,cbij -> abij', fprime[v, v], t_amp, optimize=True)
    t3 = np.einsum('ki,abkj -> abij', fprime[o, o], t_amp, optimize=True)
    t_amp = t1 + t2 - t2.transpose((1, 0, 2, 3)) \
            - t3 + t3.transpose((0, 1, 3, 2))
    
    # Divide by a 4D tensor of orbital energies
    t_amp /= (- eps[v, x, x, x] - eps[x, v, x, x] +
              eps[x, x, o, x] + eps[x, x, x, o])
   
    # Build one particle density matrix
    opdm_corr[v, v] = (1/2)*np.einsum('ijac,bcij -> ba', t_amp.T, t_amp, optimize=True)
    opdm_corr[o, o] = -(1/2)*np.einsum('jkab,abik -> ji', t_amp.T, t_amp, optimize=True)
    opdm = opdm_corr + opdm_ref 

    # Build two particle density matrix
    tpdm_corr[v, v, o, o] = t_amp
    tpdm_corr[o, o, v, v] = t_amp.T
    tpdm2 = np.einsum('rp,sq -> rspq', opdm_corr, opdm_ref, optimize=True)
    tpdm3 = np.einsum('rp,sq->rspq', opdm_ref, opdm_ref, optimize=True)
    tpdm = tpdm_corr \
        + tpdm2 - tpdm2.transpose((1, 0, 2, 3)) \
        - tpdm2.transpose((0, 1, 3, 2)) + tpdm2.transpose((1, 0, 3, 2)) \
        + tpdm3 - tpdm3.transpose((1, 0, 2, 3))

    # Newton-Raphson step
    F = np.einsum('pr,rq->pq', hmo, opdm) + (1/2) * np.einsum('prst,stqr -> pq', gmo, tpdm, optimize=True)
    X[v, o] = ((F - F.T)[v, o])/(- eps[v, x] + eps[x, o])

    # Build Newton-Raphson orbital rotation matrix
    U = la.expm(X - X.T)

    # Rotate spin-orbital coefficients
    C = C.dot(U)

    # Transform one and two electron integrals using new C
    hmo = ao_to_mo(hao, C)
    gmo = ao_to_mo_tei(gao, C)

    # Compute the energy
    E_OMP2 = E_nuc + np.einsum('pq,qp ->', hmo, opdm, optimize=True) + \
             (1/4)*np.einsum('pqrs,rspq ->', gmo, tpdm, optimize=True)
    print('OMP2 iteration: %3d Energy: %15.8f dE: %2.5E' % (iteration, E_OMP2, (E_OMP2-E_OMP2_old)))

    if (abs(E_OMP2-E_OMP2_old)) < E_conv:
        break

    # Updating values
    E_OMP2_old = E_OMP2
