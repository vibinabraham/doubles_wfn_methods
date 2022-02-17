import numpy as np
import math
from math import factorial
import itertools as it
import copy as cp
from numpy import linalg as LA
from code import *

from sys import argv

np.set_printoptions(precision=5, linewidth=200, suppress=True)

def run_scf(S, H, g, nel):
# {{{
    e_conv = 1E-8
    d_conv = 1E-8

    sal, svec = np.linalg.eigh(S)
    idx = sal.argsort()[::-1]
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal**-0.5
    sal = np.diagflat(sal)
    X = svec @ sal @ svec.T

    Fp = X.T @ H @ X
    eorb, CP = np.linalg.eigh(Fp)
    #print(CP)
    C = X @ CP
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

    #Check D
    l = np.sum(S * D)
    #print(l)

    E_old = 0.0

    print("Iter     Electronic Energy     E_conv            RMS D")
    print("----------------------------------------------------------")
    for iteration in range(2000):
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)
        #K = np.einsum("prsq,rs->pq",g,D)
        G = 2.0 * J - K
        F = H + G

        E_electric = np.sum((2.0 * H + G) * D)

        E_diff = E_electric - E_old
        E_old = E_electric

        if (abs(E_diff) < e_conv) and dconv < d_conv:
            break

        Fp = X.T @ F @ X
        Eorb, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        Cocc = C[:, :nel]
        Dold = D
        D = Cocc @ Cocc.T
        D = 0.9 * Dold + 0.1 * D
        dconv = np.sqrt(np.sum(np.square(Dold - D)))
        print("%6d %16.8f %16.8f %16.8f" %(iteration,E_electric,E_diff,dconv))
        
    print(C)
    print(F)
    print(H)
    print("CORE             %16.10f" %np.sum(2*H * D))
    print("COULOUMB         %16.10f" %np.sum(2*J * D))
    print("EXCHANGE         %16.10f" %np.sum(- K * D))
    print("Electronic Energy = %16.12f" % (E_electric))
    print()
    return E_electric, C, Eorb
# }}}

def PPP_MN_eri(U,R):
# {{{
    ## mataga nishimoto parametrization
    #atomic uints. takes in U in eV and spits out V in ev
    U = U * 0.0367493
    R =  R/.529177249
    print(R)
    V = 1/(R+(1/U))
    V = V * 27.2114


    return V
# }}}

def PPP_Ohno_eri(U,R):
# {{{
    ## Ohno parametrization
    V = 1/np.sqrt((R**2+(1/U)**2))
    return V
# }}}

def get_ppp_params_ncene(n_cene,beta,U,dist):
# {{{
    #gets the interactions for linear acene
    #n_cene 1 for benzene 2 for napthalene
    #indexing runs such that the numbering is symmetric


    n_site = 2 + n_cene * 4
    t = np.zeros((n_site,n_site))

    for i in range(0,n_site//2):
        t[i,i+1] = 1 
        t[i+1,i] = 1 
        t[n_site-i-1,n_site-i-2] = 1 
        t[n_site-i-2,n_site-i-1] = 1 
        if i % 2 == 0:
            t[i,n_site-i-1] = 1
            t[n_site-i-1,i] = 1

    h_local = -beta  * t 


    V = np.zeros((6,6))
    for i in range(0,dist.shape[0]):
        for j in range(0,dist.shape[0]):
            V[i,j] = PPP_MN_eri(U,dist[i,j])

    g_local = np.zeros((n_site,n_site,n_site,n_site))
    for i in range(0,n_site):
        for j in range(0,n_site):
            g_local[i,i,j,j] = V[i,j]
            
    return n_site,t,h_local,g_local
    # }}}

def get_hubbard_params_ncene(n_cene,beta,U):
# {{{
    #gets the interactions for linear acene
    #n_cene 1 for benzene 2 for napthalene
    #indexing runs such that the numbering is symmetric


    n_site = 2 + n_cene * 4
    t = np.zeros((n_site,n_site))

    for i in range(0,n_site//2):
        t[i,i+1] = 1 
        t[i+1,i] = 1 
        t[n_site-i-1,n_site-i-2] = 1 
        t[n_site-i-2,n_site-i-1] = 1 
        if i % 2 == 0:
            t[i,n_site-i-1] = 1
            t[n_site-i-1,i] = 1

    h_local = -beta  * t 

    g_local = np.zeros((n_site,n_site,n_site,n_site))
    for i in range(0,n_site):
        g_local[i,i,i,i] = U
            
    return n_site,t,h_local,g_local
    # }}}

def run_fci(h_active,g_active,n_orb, n_a, n_b):
    # {{{
    print()
    print(" ---------------------------------------------------------")
    print("              Full Configuration Interaction")
    print(" ---------------------------------------------------------")
    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_fci = dim_a * dim_b

    print("  Number of Orbitals         :%10d" %n_orb)
    print("  Number of a electrons      :%10d" %n_a)
    print("  Number of b electrons      :%10d" %n_b)
    print("  Full CI dimension          :%10d" %dim_fci)
    print("  Req Memory (MB)            :%10d" %(dim_fci*dim_fci*8/(1024*1024)))

    #####STORED ALL THE BINOMIAL INDEX 
    nCr_a = np.zeros((n_orb, n_orb))

    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_a[i, j] = int(nCr(i, j))

    nCr_b = np.zeros((n_orb, n_orb))

    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_b[i, j] = int(nCr(i, j))

    Det = np.zeros((dim_fci, dim_fci))

    a_index = [i for i in range(n_a)]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1
    for aa in range(0,dim_a):

        for i in range(0,n_orb):
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)


        b_index = [i for i in range(n_b)]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            for i in range(0,n_orb):
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            ebindex = []
            for i in range(0,n_orb):
                if i not in b_index:
                    ebindex.append(i)

            #TYPE: A  
            #Diagonal Terms (equations from Deadwood paper) Eqn 3.3 


            #alpha alpha string
            for i in a_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_active[i,i]
                for j in a_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_active[i,i,j,j] -  g_active[i,j,i,j])

            #beta beta string
            for i in b_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_active[i,i]
                for j in b_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_active[i,i,j,j] -  g_active[i,j,i,j])

            #alpha beta string
            for i in a_index:
                for j in b_index:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += g_active[j,j,i,i]


            #print(a_site,b_site)


            #TYPE: B  
            #single alpha (equations from Deadwood paper) Eqn 3.8 

            for j in range(0,n_orb):
                for k in range(0, j):

                    if a_site[j] != a_site[k]:

                        asite2[j], asite2[k] = asite2[k], asite2[j]

                        aindex2 = []
                        for l in range(0,n_orb):
                            if asite2[l] == 1:
                                aindex2.append(l)

                        ###Fermionic anti-symmetry
                        sym = 0
                        for l in range(k+1,j):
                            if a_site[l] == 1:
                                sym += 1
                                
                        Sphase = (-1)**sym

                        Ijaa = get_index(nCr_a, aindex2, n_orb, n_a)


                        mel =  h_active[j,k]

                        for i in a_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k]) 

                        for i in b_index:
                            mel +=  g_active[i,i,j,k] 

                        Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  mel * Sphase
                        #Det[Ijaa * dim_b + Ikbb, Ikaa * dim_b + Ikbb] += 0.5 * mel * Sphase

                        
                        #print(a_site,b_site,asite2)
                        #print(Sphase)
                        asite2 = cp.deepcopy(a_site) #imp

            #TYPE: C  
            #single beta (equations from Deadwood paper) Eqn 3.9 

            for j in range(0,n_orb):
                for k in range(0, j):

                    if b_site[j] != b_site[k]:

                        bsite2[j], bsite2[k] = bsite2[k], bsite2[j]

                        bindex2 = []
                        for l in range(0,n_orb):
                            if bsite2[l] == 1:
                                bindex2.append(l)

                        ###Fermionic anti-symmetry
                        sym = 0
                        for l in range(k+1,j):
                            if b_site[l] == 1:
                                sym += 1
                                
                        Sphase = (-1)**sym

                        Ijbb = get_index(nCr_b, bindex2, n_orb, n_b)


                        mel =  h_active[j,k]

                        for i in b_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k]) 

                        for i in a_index:
                            mel +=  g_active[i,i,j,k] 

                        Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  mel * Sphase
                        #Det[Ikaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] += 0.5 * mel * Sphase

                            
                        #print(a_site,b_site,bsite2)
                        bsite2 = cp.deepcopy(b_site) #imp

            #TYPE: D
            #Double excitation in alpha string Eqn 3.15
            for j in a_index:
                for k in a_index:
                    for l in eaindex:
                        for m in eaindex:
                            if j > k and l > m :
                                asite2[j], asite2[l] = asite2[l], asite2[j]

                                ###Fermionic anti-symmetry
                                sym1 = 0
                                for i in range(min(j,l)+1, max(j,l)):
                                    if asite2[i] == 1:
                                        sym1 += 1
                                Sphase1 = (-1)**sym1


                                asite2[k], asite2[m] = asite2[m], asite2[k]

                                ###Fermionic anti-symmetry
                                sym2 = 0
                                for i in range(min(k,m)+1, max(k,m)):
                                    if asite2[i] == 1:
                                        sym2 += 1

                                Sphase2 = (-1)**sym2


                                aindex2 = []
                                for i in range(0,n_orb):
                                    if asite2[i] == 1:
                                        aindex2.append(i)


                                Ijaa = get_index(nCr_a, aindex2, n_orb, n_a)

                                Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

                                #print("a",j,k,l,m,a_site,asite2,sym1,sym2)
                                asite2 = cp.deepcopy(a_site) #imp

            #TYPE: E
            #Double excitation in beta string Eqn 3.15
            for j in b_index:
                for k in b_index:
                    for l in ebindex:
                        for m in ebindex:
                            if j < k and l < m :
                                bsite2[j], bsite2[l] = bsite2[l], bsite2[j]

                                ###Fermionic anti-symmetry
                                sym1 = 0
                                for i in range(min(j,l)+1, max(j,l)):
                                    if bsite2[i] == 1:
                                        sym1 += 1
                                Sphase1 = (-1)**sym1

                                bsite2[k], bsite2[m] = bsite2[m], bsite2[k]
                                ###Fermionic anti-symmetry
                                sym2 = 0
                                for i in range(min(k,m)+1, max(k,m)):
                                    if bsite2[i] == 1:
                                        sym2 += 1

                                Sphase2 = (-1)**sym2



                                bindex2 = []
                                for i in range(0,n_orb):
                                    if bsite2[i] == 1:
                                        bindex2.append(i)

                                Ijbb = get_index(nCr_b, bindex2, n_orb, n_b)

                                Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

                                #print("b",b_site,bsite2)
                                bsite2 = cp.deepcopy(b_site) #imp


            #TYPE: F
            #Single alpha Single beta Eqn 3.19

            for j in range(0,n_orb):
                for k in range(0, j):
                    if a_site[j] != a_site[k]:

                        asite2[j], asite2[k] = asite2[k], asite2[j]

                        aindex2 = []
                        for l in range(0,n_orb):
                            if asite2[l] == 1:
                                aindex2.append(l)

                        ###Fermionic anti-symmetry
                        sym = 0
                        for l in range(k+1,j):
                            if a_site[l] == 1:
                                sym += 1
                                
                        aSphase = (-1)**sym

                        Ijaa = get_index(nCr_a, aindex2, n_orb, n_a)


                        for l in range(0,n_orb):
                            for m in range(0, l):
                                if b_site[l] != b_site[m]:

                                    bsite2[l], bsite2[m] = bsite2[m], bsite2[l]


                                    bindex2 = []
                                    for n in range(0,n_orb):
                                        if bsite2[n] == 1:
                                            bindex2.append(n)

                                    ###Fermionic anti-symmetry
                                    sym = 0
                                    for n in range(m+1,l):
                                        if b_site[n] == 1:
                                            sym += 1
                                            
                                    bSphase = (-1)**sym

                                    Ijbb = get_index(nCr_b, bindex2, n_orb, n_b)


                                    Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ijbb] +=   g_active[j,k,l,m] * aSphase * bSphase
                                    #Det[Ijaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] +=  0.5 * g[j,k,l,m] * aSphase * bSphase
                                    bsite2 = cp.deepcopy(b_site) #imp

                        #print(a_site,asite2)
                        asite2 = cp.deepcopy(a_site) #imp


            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp


        next_index(a_index, n_orb, n_a) #imp


    return Det
    # }}}

def nCr(n, r):
    #{{{
    if n<r:
        return 0
    else:
        return factorial(n) // factorial(r) // factorial(n-r)
    #}}}

def get_index(M, a_index, n, k):
    #{{{
    ind = 0
    if n == k :
        return ind
    for i in range(0, k):
        ind += M[int(a_index[i]), i + 1]
    return int(ind)
    #}}}

def next_index(a_index, n, k):
    #{{{
    if len(a_index) == 0:
        return False
    if a_index[0] == n - k:
        return False

    pivot = 0

    for i in range(0, k - 1):
        if a_index[i + 1] - a_index[i] >= 2:
            a_index[i] += 1
            pivot += 1
            for j in range(0, i):
                a_index[j] = j
            break

    if pivot == 0:
        a_index[k - 1] += 1
        for j in range(0, k - 1):
            a_index[j] = j

    return True
    #}}}

def run_hubbard_scf(h_local,g_local,n_elec,t):
# {{{
    print()
    print(" ---------------------------------------------------------")
    print("              Delocalized Mean-Field")
    print(" ---------------------------------------------------------")
    orb, C = np.linalg.eigh(h_local)
    if np.sum(h_local) == 0:
        print("why")
        orbt, C = np.linalg.eigh(t)


    print("Orbital energies:\n",orb,"\n")

    H = C.T @ h_local @ C                             

    g = np.einsum("pqrs,pl->lqrs",g_local,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)

    scf_nel = n_elec //2 #closed shell

    o = slice(0, scf_nel)
    v = slice(scf_nel, h_local.shape[0])

    Escf = 2*np.einsum('ii',H[o,o]) + 2*np.einsum('pqpq',g[o,o,o,o]) - np.einsum('ppqq',g[o,o,o,o])  

    print("Mean Field Energy        :%16.12f" % (Escf))

    print(C)

    return Escf,orb,H,g,C
# }}}

from sys import argv
filename = argv[1]

file = open(filename)
lines = file.readlines()
file.close()

for j in range(len(lines)):
	natom = len(lines)-2
	break
print("\nNumber of atoms:     ",natom)
xcoordinate = []
ycoordinate = []
zcoordinate = []
atom_type = []
dist_origin = []
new_atom_type = []
atom_num = []

connect = [[0 for i in range(natom)] for j in range(natom)]
real_atom =0
for i,line in enumerate(lines):
   if i>1 : 
      (n,x,y,z) = line.split()
      real_atom +=1
      X = float(x)
      Y = float(y)
      Z = float(z)
      xcoordinate.append(X)
      ycoordinate.append(Y)
      zcoordinate.append(Z)
      atom_type.append(n)

natom =real_atom
dist = (natom,natom)
dist = np.zeros(dist)

for i in range(0,natom) :
	for j in range(0,natom) :
		dist_x = xcoordinate[i] - xcoordinate[j]
		dist_y = ycoordinate[i] - ycoordinate[j]
		dist_z = zcoordinate[i] - zcoordinate[j]
		dist[i][j]   = math.sqrt(dist_x**2+dist_y**2+dist_z**2)



n_cene =1
n_site =6
R = 1.4
U = 10.84
#U = 5
beta =  2.0

n_site,t,h_local,g_local = get_ppp_params_ncene(n_cene,beta,U,dist)
n_elec = n_site
n_orb  = n_site

Escf,C, orb = run_scf(np.eye(n_site),h_local,g_local,n_elec//2)
HHH = run_fci(h_local,g_local,n_orb, 3, 3)
e0 = np.linalg.eigvalsh(HHH)
print(e0[0])
print((e0[0]- Escf)/6)


g = np.einsum("pqrs,pl->lqrs",g_local,C)
g = np.einsum("lqrs,qm->lmrs",g,C)
g = np.einsum("lmrs,rn->lmns",g,C)
g = np.einsum("lmns,so->lmno",g,C)

h = C.T @ h_local @ C

Ecc, t2 =  run_ccd_method(orb, h, g, n_elec//2,diis=False)


