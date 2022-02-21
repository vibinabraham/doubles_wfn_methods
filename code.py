import numpy as np
from math import factorial
import itertools as it
import copy as cp
import scipy.linalg as la

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

def get_hubbard_params_annulene(n_cene,beta,U):
# {{{
    #gets the interactions for linear acene
    #n_cene 1 for benzene 2 for napthalene
    #indexing runs such that the numbering is symmetric


    n_site = 2 + n_cene * 4
    t = np.zeros((n_site,n_site))

    for i in range(0,n_site-1):
        t[i,i+1] = 1 
        t[i+1,i] = 1 
    t[n_site-1,0] = 1 
    t[0,n_site-1] = 1 

    h_local = -beta  * t 

    g_local = np.zeros((n_site,n_site,n_site,n_site))
    for i in range(0,n_site):
        g_local[i,i,i,i] = U
            
    return n_site,t,h_local,g_local
    # }}}

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
        damp = True, damp_ratio=0.8, alpha=1.0,beta=1.0,gamma=1.0, reg = False, kappa = 1000.0):

    """
    Regularization possible with:
        kappa
        tikhonov
    """
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
    conv = False

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
                                                
    Escf = 2*np.einsum('pp',hocc,optimize=True) + 2*np.einsum('pqpq',gocc,optimize=True) - np.einsum('pqqp',gocc,optimize=True)    
    #print(Escf)


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
        if method2 == "singlet":
            t2 = (t2 + np.einsum('ijab->ijba',t2,optimize=True))/2 ##SINGLET CCD0


    t2_pehla = cp.deepcopy(t2)

    ### CCD Equations 

    E_cc = 2 * np.einsum('ijab,ijab',t2,gov,optimize=True) - np.einsum('ijab,ijba',t2,gov,optimize=True) 
    E_mp2 = E_cc

    #print(" SCF Energy      = %16.12f" % (Escf))
    print(" MP2 Energy corr = %16.12f" % (E_cc))
    #print(" MP2 Energy      = %16.12f" % (Escf + E_cc))
    print()
    print("   Iter       CC Energy        E_conv            RMS D")
    print(" ---------------------------------------------------------")

    max_diis = 5
    diis_vals_t2 = [t2.copy()]
    diis_errors = []
    diis_size = 0


    for cc_iter in range(0,max_iter):

        Fvv = -2 * np.einsum('mnaf,mnef-> ae',t2,gov,optimize=True) + np.einsum('mnaf,nmef-> ae',t2,gov,optimize=True)
        Fvv = 0 * Fvv

        Foo =  2 * np.einsum('inef,mnef-> mi',t2,gov,optimize=True) - np.einsum('inef,mnfe-> mi',t2,gov,optimize=True)
        Foo = 0 * Foo

        Woooo = gocc #+ 0.5 * np.einsum('ijef,mnef-> mnij',t2,gov)

        Wvvvv = gvir #+ 0.5 * np.einsum('mnab,mnef-> abef',t2,gov)

        Wovvo =  govvo #- 0.5 * np.einsum('jnfb,mnef-> mbej',t2,gov) +  np.einsum('njfb,mnef-> mbej',t2,gov) - 0.5 * np.einsum('njfb,nmef-> mbej',t2,gov)

        Wovov = -govov #+ 0.5 * np.einsum('jnfb,nmef-> mbje',t2,gov)

        Woovv = gov #+ 0.5 * np.einsum('jnfb,nmef-> mbje',t2,gov)

        #Rijab  =  gov
        Rijab = cp.deepcopy(gov)

        Rijab  +=  np.einsum('ijae,be -> ijab',t2,Fvv,optimize=True)
        Rijab  +=  np.einsum('jibe,ae -> ijab',t2,Fvv,optimize=True)
        Rijab  -=  np.einsum('imab,mj -> ijab',t2,Foo,optimize=True)
        Rijab  -=  np.einsum('mjab,mi -> ijab',t2,Foo,optimize=True)

        # linear in T2 terms
        # hole-hole ladder
        L1_ijab   =  0.5 * np.einsum('mnab,mnij -> ijab',t2,Woooo,optimize=True)
        L1_ijab  +=  0.5 * np.einsum('nmab,nmij -> ijab',t2,Woooo,optimize=True)

        # particle-particle ladder
        L2_ijab   =  0.5 * np.einsum('ijef,abef -> ijab',t2,Wvvvv,optimize=True)
        L2_ijab  +=  0.5 * np.einsum('ijfe,abfe -> ijab',t2,Wvvvv,optimize=True)


        # particle-hole ring
        L3_ijab   =  np.einsum('imae,mbej -> ijab',t2,Wovvo,optimize=True)
        L3_ijab  -=  np.einsum('miae,mbej -> ijab',t2,Wovvo,optimize=True)

        L3_ijab  +=  np.einsum('imae,mbej -> ijab',t2,Wovvo,optimize=True)
        L3_ijab  +=  np.einsum('imae,mbje -> ijab',t2,Wovov,optimize=True)

        L3_ijab  +=  np.einsum('mibe,maje -> ijab',t2,Wovov,optimize=True)
        L3_ijab  +=  np.einsum('mjae,mbie -> ijab',t2,Wovov,optimize=True)

        L3_ijab  +=  np.einsum('jmbe,maei -> ijab',t2,Wovvo,optimize=True)
        L3_ijab  -=  np.einsum('mjbe,maei -> ijab',t2,Wovvo,optimize=True)

        L3_ijab  +=  np.einsum('jmbe,maei -> ijab',t2,Wovvo,optimize=True)
        L3_ijab  +=  np.einsum('jmbe,maie -> ijab',t2,Wovov,optimize=True)

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
        DCD_1C  =  0.5 * 2 * 2 * np.einsum('klcd,ilad,kjcb -> ijab',Woovv,t2,t2,optimize=True)
        DCD_1C += -0.5     * 2 * np.einsum('klcd,ilda,kjcb -> ijab',Woovv,t2,t2,optimize=True)
        DCD_1C += -0.5     * 2 * np.einsum('klcd,ilad,kjbc -> ijab',Woovv,t2,t2,optimize=True)
        DCD_1C +=  0.5         * np.einsum('klcd,ilda,kjbc -> ijab',Woovv,t2,t2,optimize=True)

        # Ring diagram, Exchange part
        DCD_1X  = -0.5 *     2 * np.einsum('kldc,ilad,kjcb -> ijab',Woovv,t2,t2,optimize=True)
        DCD_1X +=  0.5         * np.einsum('kldc,ilad,kjbc -> ijab',Woovv,t2,t2,optimize=True)

        DCD_1X +=  0.25*     2 * np.einsum('kldc,ilda,kjcb -> ijab',Woovv,t2,t2,optimize=True)
        DCD_1X += -0.25        * np.einsum('kldc,ilda,kjbc -> ijab',Woovv,t2,t2,optimize=True)

        DCD_1X +=  0.25        * np.einsum('kldc,ilda,kjbc -> ijab',Woovv,t2,t2,optimize=True)
        DCD_1X +=  0.5         * np.einsum('kldc,ildb,kjac -> ijab',Woovv,t2,t2,optimize=True)


        # First mixed ring-ladder term
        DCD_4   = -2           * np.einsum('klcd,ilcd,kjab -> ijab',Woovv,t2,t2,optimize=True)
        DCD_4  +=                np.einsum('kldc,ilcd,kjab -> ijab',Woovv,t2,t2,optimize=True)


        # Second mixed ring-ladder term
        DCD_3   = -2           * np.einsum('klcd,klad,ijcb -> ijab',Woovv,t2,t2,optimize=True)
        DCD_3  +=                np.einsum('lkcd,klad,ijcb -> ijab',Woovv,t2,t2,optimize=True)

        # Pure ladder term
        DCD_5   =  0.5         * np.einsum('mnef,ijef,mnab -> ijab',Woovv,t2,t2,optimize=True)


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
            
            #beta = 1 alpha = 0  gamma = 0 is DCD

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
            L3_drCCD   =  2.0         * np.einsum('jcbk,ikac -> ijab',Wovvo,t2,optimize=True)
            Q_DCD_1C   =  0.5 * 2 * 2 * np.einsum('klcd,ilad,kjcb -> ijab',Woovv,t2,t2,optimize=True)

            temp = L3_drCCD + Q_DCD_1C  

        elif method == 'LCCD':
            temp = 0 * DCD_1C 


        elif method == 'CID':
            temp = 0 * DCD_1C


        temp2 = temp.swapaxes(0,1)
        temp3 = temp  + temp2.swapaxes(2,3)

        if method == 'CID':
            temp3 = - E_cc * t2 


        #this should go to zero at convergence
        #print((Rijab + temp3)-t2*Dijab)

        
        # Check if we are using regularizers        
        if (reg == False):
            t2_new = (Rijab + temp3)/Dijab
        elif reg == 'kappa':
            # if using kappa regularization
            #Kijab = (1 - np.exp(kappa*Dijab[i,j,a-occ,b-occ]))**2
            Kijab = np.ones((occ,occ,vir,vir))
            Kijab = (Kijab - np.exp(kappa*Dijab))**2
            t2_new = ((Rijab + temp3)/Dijab) * Kijab
        elif reg == 'tikhonov':
            t2_new = (Rijab + temp3)*(Dijab/np.square(Dijab)+kappa)


        #print(np.max(Dijab))

        #t2_new = (Rijab + DCD_1C)/Dijab
        #t2_new = (Rijab + DCD_1X)/Dijab
        #t2_new = (Rijab)/Dijab
        

        if method2 == "normal":
            t2_new = t2_new
        elif method2 == "singlet":
            t2_new = (t2_new + np.einsum('ijab->ijba',t2_new,optimize=True))/2 ##SINGLET CCD0
            
        elif method2 == "pairsinglet":
            t2_new = (t2_new + np.einsum('ijab->ijba',t2_new,optimize=True))/2 ##SINGLET CCD0
            # Pair CCD or pair CID is a drastic approximation to singlet CCD/singlet CID 
            # Need to zero all off-diagonal elements
            for i in range(0,occ):
                for j in range(0,occ):
                    for a in range(0,vir):
                        for b in range(0,vir):
                            if a !=b or i !=j:
                                t2_new[i,j,a,b] = 0.0
            #t2_new = t2_new.swapaxes(1,2)
            #print(t2_new[0,0,:,:])
            #print(t2_new[1,1,:,:])
            #print(t2_new[2,2,:,:])
            #print(t2_new[3,3,:,:])
            #print(t2_new[4,4,:,:])
            #print(t2_new[:,:,0,0])
            #print(t2_new[:,:,1,1])
            #print(t2_new[:,:,2,2])
            #print(t2_new[:,:,3,3])
            #print(t2_new.shape)
            #t2_new.shape = (occ*vir*occ*vir)
            #print(t2_new)


        elif method2 == "triplet":
            t2_new = (t2_new - np.einsum('ijab->ijba',t2_new,optimize=True))/2 ##TRIPLET CCD0


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


        E_cc_new = 2 * np.einsum('ijab,ijab',t2_new,gov,optimize=True) - np.einsum('ijab,ijba',t2_new,gov,optimize=True) 
        if method == 'directringCCD' :
            # E1 =    TA::dot(g_abij("a,b,i,j") + r2("a,b,i,j"), 2 * t2("a,b,i,j"));
            # E_SOSEX = TA::dot(g_abij("a,b,i,j") + r2("a,b,i,j"), 2*t2("a,b,i,j") - t2("a,b,j,i"));
            E_cc_new = 2 * np.einsum('ijab,ijab',t2_new,gov,optimize=True)    


        E_diff = abs(E_cc - E_cc_new)

        drms = np.sqrt(np.sum(np.square(t2_new - t2)))/(nCr(occ,2)*nCr(vir,2))
        print("%6d %16.8f %16.8f %16.6f" %(cc_iter,E_cc,E_diff,drms))
        if (abs(E_diff) < e_conv) and drms < d_conv:
            conv = True
            break
        elif cc_iter == max_iter-1:
            print(" CCD did not converge")
        E_cc = E_cc_new
        

        if damp:
            t2 = damp_ratio * t2 + (1 - damp_ratio) * t2_new
        else:
            t2 = t2_new


    print("\nCCD Total       Energy  :%16.12f" % (Escf+E_cc))
    print("\nCCD Correlation Energy  :%16.12f" % (E_cc))
    if (method == 'pCCD' or method == 'pDCD') : 
            print(" You just did a paramterized CCD/DCD calculation with parameters" )
            print(" alpha       = %16.12f" % (alpha))
            print(" beta        = %16.12f" % (beta))
            print(" gamma       = %16.12f" % (gamma))
    #print("\n CCD Energy/site          :%16.12f" % (E_cc/tot))
    print(method)
    print(method2)

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
    #print("MP2:%16.8f"%(Escf+Emp2))
    return Emp2
# }}}

def get_hubbard_params(n_site,beta,U,pbc=True):
# {{{
    #gets the interactions for linear hubbard
    print(" ---------------------------------------------------------")
    print("                       Hubbard model")
    print(" ---------------------------------------------------------")
    print(" nsite:%6d"%n_site)
    print(" beta :%10.6f"%beta)
    print(" U    :%10.6f"%U)

    t = np.zeros((n_site,n_site))

    for i in range(0,n_site-1):
        t[i,i+1] = 1 
        t[i+1,i] = 1 
    if pbc:
        t[n_site-1,0] = 1 
        t[0,n_site-1] = 1 

    h_local = -beta  * t 

    g_local = np.zeros((n_site,n_site,n_site,n_site))
    for i in range(0,n_site):
        g_local[i,i,i,i] = U
            
    return h_local,g_local
    # }}}


##############
#
# Testing kappa MP2 and xBW https://aip.scitation.org/doi/10.1063/5.0078119
#
##############

def run_bw(orb,H,g,closed_shell_nel,level_shift='xbw'):
    """
    shift with BW method
    level_shit options: 
        bw:  will do a BW2 calculation (not size extensive)
        xbw: will do johannes shift of Ec/2*closed_shell_nel
        mp:  will do MP2 calculation
        value:  user can give a constant value

    """
# {{{
    print()
    print(" ---------------------------------------------------------")
    print("                         BW         ")
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
    
    Ec = 0
    for it in range(0,30):

        if level_shift == 'xbw':
            shift = Ec/(2*nel)
        if level_shift == 'mp':
            shift = 0
        if level_shift == 'bw':
            shift = Ec
        if level_shift == 'value':
            shift =  0.4

        Dijab = np.zeros((occ,occ,vir,vir))
        Rijab = np.zeros((occ,occ,vir,vir))
        t2 = np.zeros((occ,occ,vir,vir))
        for i in range(0,nel):
           for j in range(0,nel):
              for a in range(nel,tot):
                 for b in range(nel,tot):
                    Dijab[i,j,a-occ,b-occ] = (orb[i] + orb[j] - orb[a] - orb[b] + shift )
                    t2[i,j,a-occ,b-occ] = gov[i,j,a-occ,b-occ]/Dijab[i,j,a-occ,b-occ]
        Ec_old = Ec
        Ec = 2 * np.einsum('ijab,ijab',t2,gov) - np.einsum('ijab,ijba',t2,gov) 
        print("%4d %12.6f"%(it,Ec))
        if abs(Ec - Ec_old) < 1e-5:
            print("Converged")
            print("Etot:%16.8f method:%4s"%(Ec+Escf,level_shift))
            break 
    return Ec
# }}}

def run_kmp2(orb,H,g,closed_shell_nel,kappa = 0.4):
    """
    kappa MP2  method. Cant do oo yet since there will be more values in gradient. 
    """
# {{{
    print()
    print(" ---------------------------------------------------------")
    print("                         kMP2         ")
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
    t2 = np.zeros((occ,occ,vir,vir))
    Dijab = np.zeros((occ,occ,vir,vir))
    Kappa = np.zeros((occ,occ,vir,vir))
    for i in range(0,nel):
       for j in range(0,nel):
          for a in range(nel,tot):
             for b in range(nel,tot):
                Dijab[i,j,a-occ,b-occ] = (orb[i] + orb[j] - orb[a] - orb[b])
                Kijab = (1 - np.exp(kappa*Dijab[i,j,a-occ,b-occ]))**2
                t2[i,j,a-occ,b-occ] = (gov[i,j,a-occ,b-occ]/Dijab[i,j,a-occ,b-occ]) * Kijab
    
    Ec = 2 * np.einsum('ijab,ijab',t2,gov) - np.einsum('ijab,ijba',t2,gov) 
    print("Ekmp2:%16.8f  kappa:%12.4f"%(Ec+Escf,kappa))
    return Ec
# }}}

##############
#
# For orbital opt of MP2
#
##############

def spatial_2_spin(eps,hao,gao,C):
# {{{
    '''
    Spin block 1e integrals
    Spin blocks 2-electron integrals
    Using np.kron, we project I and I tranpose into the space of the 2x2 ide
    The result is our 2-electron integral tensor in spin orbital notation
    '''

    # ==> Orbital Energies <==
    eps = np.append(eps, eps)

    # ==> core Hamiltoniam <==

    # Using np.kron, we project h into the space of the 2x2 identity
    # The result is the core Hamiltonian in the spin orbital formulation
    hao = np.kron(np.eye(2), hao)

    # ==> 2e integral <==
    identity = np.eye(2)
    gao = np.kron(identity, gao)
    gao_spinblock =  np.kron(identity, gao.T)

    # Convert chemist's notation to physicist's notation, and antisymmetrize
    # (pq | rs) ---> <pr | qs>
    # <pr||qs> = <pr | qs> - <pr | sq>
    gao = gao_spinblock.transpose(0, 2, 1, 3) - gao_spinblock.transpose(0, 2, 3, 1)



    # ==> MO Coefficients <==
    # Get coefficients, block, and sort
    Ca = np.asarray(C)
    Cb = np.asarray(C)
    C = np.block([
                 [      Ca,         np.zeros_like(Cb)],
                 [np.zeros_like(Ca),          Cb     ]])

    #print(C)
    # Sort the columns of C according to the order of orbital energies
    C = C[:, eps.argsort()]
    #print(C)

    return eps, hao, gao, C
# }}}

def ao_to_mo(hao, gao, C):
# {{{
    '''
    Transform hao, which is the core Hamiltonian in the spin orbital basis,
    into the MO basis using MO coefficients

    Transform gao, which is the spin-blocked 4d array of physicist's notation,
    antisymmetric two-electron integrals, into the MO basis using MO coefficients
    '''
    
    # ==> AO to MO transformation functions <==
    hmo =  np.einsum('pQ, pP -> PQ', 
           np.einsum('pq, qQ -> pQ', hao, C, optimize=True), C, optimize=True)

    gmo =  np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS', 
           np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)
    return hmo, gmo
# }}}

def oomp2(eps,hmo,gmo,C,nocc,ecore,maxiter=40,E_conv=1e-8):
    """
    Optimize orbitals 
    """
# {{{
    nso = hmo.shape[0]
    nvirt = nso - nocc         # Number of virtual orbitals

    # code block 11
    # Make slices
    o = slice(None, nocc)
    v = slice(nocc, None)
    x = np.newaxis

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

    for iteration in range(maxiter):

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
        #hmo, gmo = ao_to_mo(hao, gao, C)
        hmo, gmo = ao_to_mo(hmo, gmo, U)

        # Compute the energy
        E_OMP2 = ecore + np.einsum('pq,qp ->', hmo, opdm, optimize=True) + \
                 (1/4)*np.einsum('pqrs,rspq ->', gmo, tpdm, optimize=True)
        print('OMP2 iteration: %3d Energy: %15.8f dE: %2.5E' % (iteration, E_OMP2, (E_OMP2-E_OMP2_old)))

        if (abs(E_OMP2-E_OMP2_old)) < E_conv:
            break

        # Updating values
        E_OMP2_old = E_OMP2
    return E_OMP2
# }}}

