import numpy as np
from math import factorial
from scipy.special import comb
import itertools as it
import copy as cp
from numpy import linalg as LA


def run_hubbard_scf(h_local,g_local,closed_shell_nel,t):
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

    h = C.T @ h_local @ C                             

    g = np.einsum("pqrs,pl->lqrs",g_local,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)


    o = slice(0, closed_shell_nel)
    v = slice(closed_shell_nel, h_local.shape[0])

    Escf = 2*np.einsum('ii',h[o,o]) + 2*np.einsum('ppqq',g[o,o,o,o]) - np.einsum('pqqp',g[o,o,o,o])

    print("Mean Field Energy        :%16.12f" % (Escf))

    print(C)

    return Escf,orb,h,g,C
# }}}

def run_ccd_method(orb, h, g, closed_shell_nel, t2 = None, 
        method = "CCD",method2="normal", 
        max_iter = 1000, e_conv = 1e-10, d_conv = 1e-8, 
        diis = True, diis_start=10, max_diis = 5,
        damp = True, damp_ratio=0.8, alpha=1.0,beta=1.0,gamma=1.0):
# {{{
    print()
    print(" ---------------------------------------------------------")
    print("                 Coupled Cluster Doubles")
    print(" ---------------------------------------------------------")
    print("                     Method  :",method)
    print(" ---------------------------------------------------------")
    print("                     Method2 :",method2)
    print(" ---------------------------------------------------------")
    print("")

    print("WARNING -transforming to physicist notation")
    g = g.swapaxes(1,2)
    nel = closed_shell_nel
    occ = nel
    tot = h.shape[0]
    vir = tot - occ

    o = slice(0, occ)
    v = slice(occ, tot)

    #T2 integrals
    hocc  = h[o,o]                         
    gocc  = g[o,o,o,o]
    gvir  = g[v,v,v,v]
    govvo = g[o,v,v,o]
    govov = g[o,v,o,v]
    gov   = g[o,o,v,v]
                                                
    Escf = 2*np.einsum('pp',hocc) + 2*np.einsum('pqpq',gocc) - np.einsum('pqqp',gocc)    
    print(Escf)


    t2    = np.zeros((occ,occ,vir,vir))
    Dijab = np.zeros((occ,occ,vir,vir))
    Rijab = np.zeros((occ,occ,vir,vir))
    for i in range(0,nel):
       for j in range(0,nel):
          for a in range(nel,tot):
             for b in range(nel,tot):
                Dijab[i,j,a-occ,b-occ] = (orb[i] + orb[j] - orb[a] - orb[b])

    if t2 is None:
        t2     = np.zeros((occ,occ,vir,vir))
        for i in range(0,nel):
           for j in range(0,nel):
              for a in range(nel,tot):
                 for b in range(nel,tot):
                    t2[i,j,a-occ,b-occ] = (g[i,j,a,b]/Dijab[i,j,a-occ,b-occ]) 
    else:
        assert(t2.shape == (occ,occ,vir,vir))

    t2_pehla = cp.deepcopy(t2)

    ### CCD Equations 

    E_cc = 2 * np.einsum('ijab,ijab',t2,gov) - np.einsum('ijab,ijba',t2,gov) 
    E_mp2 = E_cc

    print(" SCF Energy      = %16.12f" % (Escf))
    print(" MP2 Energy corr = %16.12f" % (E_cc))
    print(" MP2 Energy      = %16.12f" % (Escf + E_cc))
    print()
    print("   Iter       CC Energy        E_conv            RMS D")
    print(" ---------------------------------------------------------")

    max_diis = 5
    diis_vals_t2 = [t2.copy()]
    diis_errors = []
    diis_size = 0


    for cc_iter in range(0,max_iter):

        Fvv = -2 * np.einsum('mnaf,mnef-> ae',t2,gov) + np.einsum('mnaf,nmef-> ae',t2,gov)
        Fvv = 0 * Fvv

        Foo =  2 * np.einsum('inef,mnef-> mi',t2,gov) - np.einsum('inef,mnfe-> mi',t2,gov)
        Foo = 0 * Foo

        Woooo = gocc #+ 0.5 * np.einsum('ijef,mnef-> mnij',t2,gov)

        Wvvvv = gvir #+ 0.5 * np.einsum('mnab,mnef-> abef',t2,gov)

        Wovvo =  govvo #- 0.5 * np.einsum('jnfb,mnef-> mbej',t2,gov) +  np.einsum('njfb,mnef-> mbej',t2,gov) - 0.5 * np.einsum('njfb,nmef-> mbej',t2,gov)

        Wovov = -govov #+ 0.5 * np.einsum('jnfb,nmef-> mbje',t2,gov)

        Woovv = gov #+ 0.5 * np.einsum('jnfb,nmef-> mbje',t2,gov)

        #Rijab  =  gov
        Rijab = cp.deepcopy(gov)

        Rijab  +=  np.einsum('ijae,be -> ijab',t2,Fvv)
        Rijab  +=  np.einsum('jibe,ae -> ijab',t2,Fvv)
        Rijab  -=  np.einsum('imab,mj -> ijab',t2,Foo)
        Rijab  -=  np.einsum('mjab,mi -> ijab',t2,Foo)

        # linear in T2 terms
        # hole-hole ladder
        L1_ijab   =  0.5 * np.einsum('mnab,mnij -> ijab',t2,Woooo)
        L1_ijab  +=  0.5 * np.einsum('nmab,nmij -> ijab',t2,Woooo)

        # particle-particle ladder
        L2_ijab   =  0.5 * np.einsum('ijef,abef -> ijab',t2,Wvvvv)
        L2_ijab  +=  0.5 * np.einsum('ijfe,abfe -> ijab',t2,Wvvvv)


        # particle-hole ring
        L3_ijab   =  np.einsum('imae,mbej -> ijab',t2,Wovvo)
        L3_ijab  -=  np.einsum('miae,mbej -> ijab',t2,Wovvo)

        L3_ijab  +=  np.einsum('imae,mbej -> ijab',t2,Wovvo)
        L3_ijab  +=  np.einsum('imae,mbje -> ijab',t2,Wovov)

        L3_ijab  +=  np.einsum('mibe,maje -> ijab',t2,Wovov)
        L3_ijab  +=  np.einsum('mjae,mbie -> ijab',t2,Wovov)

        L3_ijab  +=  np.einsum('jmbe,maei -> ijab',t2,Wovvo)
        L3_ijab  -=  np.einsum('mjbe,maei -> ijab',t2,Wovvo)

        L3_ijab  +=  np.einsum('jmbe,maei -> ijab',t2,Wovvo)
        L3_ijab  +=  np.einsum('jmbe,maie -> ijab',t2,Wovov)

        if (method != 'ringCCD' and  method!= 'directringCCD' and method!= 'directringCCD+SOSEX') :
             Rijab    += L1_ijab + L2_ijab + L3_ijab
        elif (method == 'ringCCD'):
             Rijab    += L3_ijab
        elif (method == 'directringCCD' and method!= 'directringCCD+SOSEX'):
        # include just the coulomb part of this ring diagram, loss of antisymmetry in t2 amplitudes
        # t2_d_ring("a,b,i,j") = 0.5 * (2.0 * g_aijb("c,j,k,b")) * (2.0 * t2("a,c,i,k"));
        # Will add this term later with the quadratic term
             Rijab    += 0.0 * L3_ijab



        # Quadratic in T2 terms

        # Ring diagram, Coulomb part
        DCD_1C  =  0.5 * 2 * 2 * np.einsum('klcd,ilad,kjcb -> ijab',Woovv,t2,t2)
        DCD_1C += -0.5     * 2 * np.einsum('klcd,ilda,kjcb -> ijab',Woovv,t2,t2)
        DCD_1C += -0.5     * 2 * np.einsum('klcd,ilad,kjbc -> ijab',Woovv,t2,t2)
        DCD_1C +=  0.5         * np.einsum('klcd,ilda,kjbc -> ijab',Woovv,t2,t2)

        # Ring diagram, Exchange part
        DCD_1X  = -0.5 *     2 * np.einsum('kldc,ilad,kjcb -> ijab',Woovv,t2,t2)
        DCD_1X +=  0.5         * np.einsum('kldc,ilad,kjbc -> ijab',Woovv,t2,t2)

        DCD_1X +=  0.25*     2 * np.einsum('kldc,ilda,kjcb -> ijab',Woovv,t2,t2)
        DCD_1X += -0.25        * np.einsum('kldc,ilda,kjbc -> ijab',Woovv,t2,t2)

        DCD_1X +=  0.25        * np.einsum('kldc,ilda,kjbc -> ijab',Woovv,t2,t2)
        DCD_1X +=  0.5         * np.einsum('kldc,ildb,kjac -> ijab',Woovv,t2,t2)


        # First mixed ring-ladder term
        DCD_4   = -2           * np.einsum('klcd,ilcd,kjab -> ijab',Woovv,t2,t2)
        DCD_4  +=                np.einsum('kldc,ilcd,kjab -> ijab',Woovv,t2,t2)


        # Second mixed ring-ladder term
        DCD_3   = -2           * np.einsum('klcd,klad,ijcb -> ijab',Woovv,t2,t2)
        DCD_3  +=                np.einsum('lkcd,klad,ijcb -> ijab',Woovv,t2,t2)

        # Pure ladder term
        DCD_5   =  0.5         * np.einsum('mnef,ijef,mnab -> ijab',Woovv,t2,t2)


        if method == 'CCD':
            #EXACT
            temp = DCD_1C + DCD_1X + DCD_3 + DCD_4 + DCD_5

        elif method == 'DCD':
            #DCD
            temp = DCD_1C + 0.5 *(DCD_3 + DCD_4)

        elif method == 'ACPD14':
            #ACPD14
            temp = DCD_1C +  DCD_4

        elif method == 'ACPD45':
            #ACPD45
            temp = DCD_4 +  DCD_5

        elif method == 'pCCD':
            #parametrized CCD, takes two parameters alpha and beta
            # alpha=1 , beta=1 is CCD
            temp = beta * (DCD_1C + DCD_1X + DCD_3) +  \
                   alpha * ((0.5 * DCD_4) + DCD_5) + 0.5 * DCD_4

        elif method == 'pDCD':
            #This is an experiment
            #parametrized DCD, takes three parameters alpha, beta and gamma
            # alpha=1 , beta=1, gamma=1 is CCD
            temp = beta * (DCD_1C + 0.5 * DCD_3) + gamma *(DCD_1X + 0.5 * DCD_3 ) + \
                   alpha * ((0.5 * DCD_4) + DCD_5) + 0.5 * DCD_4

        elif method == 'ACPD1':
            temp = DCD_1C

        elif method == 'ringCCD':
            temp = DCD_1C + DCD_1X

        elif (method == 'directringCCD' or method == 'directringCCD+SOSEX'):
        # Include the coulomb part of the ring diagram, sacrifice antisymmetry of t2 amplitudes    
        # t2_d_ring("a,b,i,j") = 0.5 * (2.0 * g_aijb("c,j,k,b")) * (2.0 * t2("a,c,i,k"));
        # t2_dd_D_c("a,b,i,j")  = 0.5 * g_ijab("k,l,c,d") * (2.0 * t2("a,d,i,l") ) *( 2.0 * t2("c,b,k,j") );
            L3_drCCD   =  2.0         * np.einsum('jcbk,ikac -> ijab',Wovvo,t2)
            Q_DCD_1C   =  0.5 * 2 * 2 * np.einsum('klcd,ilad,kjcb -> ijab',Woovv,t2,t2)

            temp = L3_drCCD + Q_DCD_1C  

        elif method == 'LCC':
            temp = 0 * DCD_1C 


        elif method == 'CID':
            temp = 0 * DCD_1C


        temp2 = temp.swapaxes(0,1)
        temp3 = temp  + temp2.swapaxes(2,3)

        if method == 'CID':
            temp3 = - E_cc * t2 


        #this should go to zero at convergence
        #print((Rijab + temp3)-t2*Dijab)

        t2_new = (Rijab + temp3)/Dijab

        #t2_new = (Rijab + DCD_1C)/Dijab
        #t2_new = (Rijab + DCD_1X)/Dijab
        #t2_new = (Rijab)/Dijab
        

        if method2 == "normal":
            t2_new = t2_new
        if (method2 == "singlet" or method2 == "pairsinglet"):
            t2_new = (t2_new + np.einsum('ijab->ijba',t2_new))/2 ##SINGLET CCD0
            # Pair CCD or pair CID is a drastic approximation to singlet CCD/singlet CID 
            # Need to zero all off-diagonal elements
            if method2 == "pairsinglet" :
                for i in range(0,nel):
                   for j in range(0,nel):
                      for a in range(nel,tot):
                         for b in range(nel,tot):
                            if i != j or a !=b :
                               t2[i,j,a-occ,b-occ] = 0.0


        if method2 == "triplet":
            t2_new = (t2_new - np.einsum('ijab->ijba',t2_new))/2 ##TRIPLET CCD0


        #####################DIIISSSSSSSSSSSSSSSSSS
        oldt2 = t2.copy()

        diis_vals_t2.append(t2_new.copy())

        error_t2 = (t2_new - oldt2).ravel()
        diis_errors.append(error_t2)

        #error_t2 = (t2_new - oldt2)
        #diis_errors.append(error_t2)



        if diis: 
            if cc_iter >= diis_start:
                # Limit size of DIIS vector
                if (len(diis_vals_t2) > max_diis):
                    del diis_vals_t2[0]
                    del diis_errors[0]
                diis_size = len(diis_vals_t2) - 1

                # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
                B = np.ones((diis_size + 1, diis_size + 1)) * -1
                B[-1, -1] = 0

                for n1, e1 in enumerate(diis_errors):
                    for n2, e2 in enumerate(diis_errors):
                        # Vectordot the error vectors
                        B[n1, n2] = np.dot(e1, e2)
                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()


                # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
                resid = np.zeros(diis_size + 1)
                resid[-1] = -1



                #print(B)
                #print(resid)
                # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
                ci = np.linalg.solve(B, resid)

                #print(ci)
                # Calculate new amplitudes
                t2_new[:] = 0

                for num in range(diis_size):
                    t2_new += ci[num] * diis_vals_t2[num + 1]
                # End DIIS amplitude update


        E_cc_new = 2 * np.einsum('ijab,ijab',t2_new,gov) - np.einsum('ijab,ijba',t2_new,gov) 
        if method == 'directringCCD' :
            # E1 =    TA::dot(g_abij("a,b,i,j") + r2("a,b,i,j"), 2 * t2("a,b,i,j"));
            # E_SOSEX = TA::dot(g_abij("a,b,i,j") + r2("a,b,i,j"), 2*t2("a,b,i,j") - t2("a,b,j,i"));
            E_cc_new = 2 * np.einsum('ijab,ijab',t2_new,gov)    


        E_diff = abs(E_cc - E_cc_new)

        drms = np.sqrt(np.sum(np.square(t2_new - t2)))/(nCr(occ,2)*nCr(vir,2))
        print("%6d %16.8f %16.8f %16.6f" %(cc_iter,E_cc,E_diff,drms))
        if (abs(E_diff) < e_conv) and drms < d_conv:
            break
        elif cc_iter == max_iter-1:
            print(" CCD did not converge")
        E_cc = E_cc_new
        

        if damp:
            t2 = damp_ratio * t2 + (1 - damp_ratio) * t2_new
        else:
            t2 = t2_new


    print("\nCCD Energy               :%16.12f" % (Escf+E_cc))
    if (method == 'pCCD' or method == 'pDCD') : 
            print(" You just did a paramterized CCD/DCD calculation with parameters" )
            print(" alpha       = %16.12f" % (alpha))
            print(" beta        = %16.12f" % (beta))
            print(" gamma       = %16.12f" % (gamma))
    #print("\n CCD Energy/site          :%16.12f" % (E_cc/tot))

    return E_cc,t2_new
# }}}

def nCr(n, r):
    #{{{
    if n<r:
        return 0
    else:
        return factorial(n) // factorial(r) // factorial(n-r)
    #}}}

def run_mp2(orb,H,g,closed_shell_nel):
# {{{
    print()
    print(" ---------------------------------------------------------")
    print("                         MP2         ")
    print(" ---------------------------------------------------------")

    print("WARNING -transforming to physicist notation")
    g = g.swapaxes(1,2)
    nel = closed_shell_nel
    occ = nel
    tot = H.shape[0]
    vir = tot - occ



    Hocc = H[:nel,:nel]                         
    gocc = g[:nel,:nel,:nel,:nel]               
    gvir = g[nel:tot,nel:tot,nel:tot,nel:tot]   
    gov = g[:nel,:nel,nel:tot,nel:tot]         
    Escf0  =  2*np.einsum('ii',Hocc)            
    Escf1  =  2*np.einsum('pqpq',gocc)          
    Escf2  =  np.einsum('ppqq',gocc)            
                                                
    Escf = Escf0 + Escf1 - Escf2                

    #Guess T2 amplitudes, if none is provided
    #t2 = np.zeros((occ,occ,vir,vir))
    
    Dijab = np.zeros((occ,occ,vir,vir))
    Rijab = np.zeros((occ,occ,vir,vir))
    t2 = np.zeros((occ,occ,vir,vir))
    for i in range(0,nel):
       for j in range(0,nel):
          for a in range(nel,tot):
             for b in range(nel,tot):
                Dijab[i,j,a-occ,b-occ] = (orb[i] + orb[j] - orb[a] - orb[b])
                t2[i,j,a-occ,b-occ] = gov[i,j,a-occ,b-occ]/Dijab[i,j,a-occ,b-occ]



    Emp2 = 2 * np.einsum('ijab,ijab',t2,gov) - np.einsum('ijab,ijba',t2,gov) 
    return Emp2
# }}}

