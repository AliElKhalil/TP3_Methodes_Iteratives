# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:56:49 2020

@author: aliel
"""
########################################################## Préliminaires
import numpy as np 
import random
import time as time
import matplotlib.pyplot as plt  #importation des bibliothèque utilisées

def Decompose(A):       #Fonction qui décompose une matrice A=D-E-F
    n = len(A)
    D = np.zeros(shape=(n,n))
    E = np.zeros(shape=(n,n))
    F = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i,i] = A[i,i]
            if i > j:
                E[i,j] = -A[i,j]
            if i < j:
                F[i,j] = -A[i,j]
    return D,E,F

def fusion(g,d):        #fonction préliminaire pour la fonction suivante
    R = []
    i_d=0
    i_g=0
    while i_g < len(g) and i_d < len(d):        
        if g[i_g] <= d[i_d]:
            R.append(g[i_g])
            i_g += 1
        else:
            R.append(d[i_d])
            i_d += 1
    if i_g<len(g):
        R.extend(g[i_g:])
    if i_d<len(d):
        R.extend(d[i_d:])
    return R
 
def tri_fusion(K):  #tri fusion
    if len(K) <= 1:
        return K
    M = len(K) // 2
    g = K[:M]
    d = K[M:]
    g = tri_fusion(g)
    d = tri_fusion(d)
    return fusion(g, d)

def gensys(n):  #Fonction qui créer un système (A,b) avec A à diagonale dominante
    A,b = np.random.randint(-100, 100, size=(n, n)), np.random.randint(-100, 100, size=(n, 1))
    for i in range(n):
        l=0
        for j in range(n):
            l+=abs(A[i,j])
        d=random.randint(-n*100,n*100)
        if d>0:
            A[i,i]=d+l
        else :
            A[i,i]=d-l-1
    return A,b

########################################################## Partie 1

def MIGenerale(M,N,b,x0,epsilon,Nitermax): #Question 1
    Niter=0
    x=x0
    xv=x0+epsilon+1
    while Nitermax>Niter and np.linalg.norm(xv-x)>epsilon:
        xv=x
        x=np.linalg.solve(M,b+np.dot(N,xv))
        Niter+=1
    return x,Niter,np.linalg.norm(xv-x)

def MIJacobi(A,b,x0,epsilon,Nitermax):  #Question 2
    D,E,F=Decompose(A)
    M = D
    N = E + F
    return MIGenerale(M,N,b,x0,epsilon,Nitermax)

def MIGaussSeidel(A,b,x0,epsilon,Nitermax): #Question 3
    D,E,F=Decompose(A)
    M = D - E
    N = F
    return MIGenerale(M,N,b,x0,epsilon,Nitermax)

def MIRelaxation(A,b,x0,epsilon,Nitermax,omega):    #Question 4     
    D,E,F=Decompose(A)
    M = (1/omega)*D - E
    N = M - A
    return MIGenerale(M,N,b,x0,epsilon,Nitermax)

def MIRichardson(A,b,x0,epsilon,Nitermax,omega):    #En bonus 
    n=len(A)
    M = (1/omega)*np.ones(shape=(n,n))
    N = M - A
    return MIGenerale(M,N,b,x0,epsilon,Nitermax)

def MIJOR(A,b,x0,epsilon,Nitermax, omega):
    D,E,F=Decompose(A)
    M = (1/omega)*D
    N = M - A
    return MIGenerale(M,N,b,x0,epsilon,Nitermax)
########################################################## Partie 2
def A1(n):
    A=np.zeros((n,n))
    for i in range (0,n):
        for j in range (0,n):
            if i==j:
                A[i,j]=3
            else:
                A[i,j]=1/(12+(3*i-5*j)**2)
    return A

def b(n):
    b=np.zeros((n,1))
    for i in range (0,n):
        b[i]=np.cos(i/8)
    return b

def A2(n):
    A=np.zeros((n,n))
    for i in range (0,n):
        for j in range (0,n):
            if i==j:
                A[i,j]=3
            else:
                A[i,j]=1/(1+3*np.abs(i-j))
    return A


def SolutionEffective(n,k,N):
    if k==1:
        A=A1(n)
    elif k==2:
        A=A2(n)
    else :
        return False
    IJ=[]
    IGS=[]
    P=[]
    x0=np.zeros((n,1))
    Nitermax=N
    NbiterJ=[]
    NbiterGS=[]
    xJ=[]
    xGS=[]
    for e in range (3,14):
        epsilon=10**(-e)
        IJ.append(MIJacobi(A,b(n),x0,epsilon,N)[2])
        NbiterJ.append(MIJacobi(A,b(n),x0,epsilon,N)[1])
        IGS.append(MIGaussSeidel(A,b(n),x0,epsilon,Nitermax)[2])
        NbiterGS.append(MIGaussSeidel(A,b(n),x0,epsilon,Nitermax)[1])
        P.append(epsilon)
        xJ.append(MIJacobi(A,b(n),x0,epsilon,N)[0])
        xGS.append(MIGaussSeidel(A,b(n),x0,epsilon,Nitermax)[0])
    PEJ=[]
    PEGS=[]
    x=np.linalg.solve(A,b(n))
    ErrJ=[]
    ErrGS=[]
    for i in range (0,len(P)):
        NJ=np.linalg.norm(IJ[i]-P[i])
        NGS=np.linalg.norm(IGS[i]-P[i])
        PEJ.append(NJ)
        PEGS.append(NGS)
        ErrJ.append(np.linalg.norm(x-xJ[i]))
        ErrGS.append(np.linalg.norm(x-xGS[i]))
    return [P,PEJ,PEGS,NbiterJ,NbiterGS,ErrJ,ErrGS]
        
def CourbeP(n,k,N): #Courbe classique des précision
    K=SolutionEffective(n,k,N)
    x=K[0]
    y1=K[1]
    y2=K[2]
    y3=K[5]
    y4=K[6]
    plt.plot(x,y1,label="Précision effective Jacobi")
    plt.plot(x,y2,label="Précision effective Gauss-Seidel")
    plt.plot(x,y3, label="Erreur réelle Jacobi")
    plt.plot(x,y4,label="Erreur réelle Gauss-Seidel")
    plt.xlabel("Précision choisie")
    plt.ylabel("Précision Effective")
    plt.title("Courbe des précisions, question "+str(k))
    plt.legend()
    plt.show()
    
def CourbelogP(n,k,N):  #Courbe en log des précision 
    K=SolutionEffective(n,k,N)
    x=K[0]
    y1=K[1]
    y2=K[2]
    y3=K[5]
    y4=K[6]
    plt.loglog(x,y1,label="Précision effective Jacobi")
    plt.loglog(x,y2,label="Précision effective Gauss-Seidel")
    plt.loglog(x,y3, label="Erreur réelle Jacobi")
    plt.loglog(x,y4,label="Erreur réelle Gauss-Seidel")
    plt.xlabel("Précision choisie")
    plt.ylabel("Précision éffective")
    plt.title("Courbe des précisions en log, question "+str(k))
    plt.legend()
    plt.show()
    
def CourbeNbIteration(n,k,N):   #Courbe classique du nombre d'itération 
    K=SolutionEffective(n,k,N)
    x=K[0]
    y1=K[3]
    y2=K[4]
    plt.plot(x,y1,label="Nombre d'itération pour Jacobi")
    plt.plot(x,y2,label="Nombre d'itération pour Gauss-Seidel")
    plt.xlabel("précision choisie")
    plt.ylabel("Nombre d'itération")
    plt.title("Nombre d'itération selon la précision, question "+str(k))
    plt.legend()
    plt.show()
    
def CourbelogNbIteration(n,k,N):    #Courbe en log sur l'axe des abcisses du nombre d'itération 
    K=SolutionEffective(n,k,N)
    x=K[0]
    y1=K[3]
    y2=K[4]
    plt.semilogx(x,y1,label="Nombre d'itération pour Jacobi")
    plt.semilogx(x,y2,label="Nombre d'itération pour Gauss-Seidel")
    plt.xlabel("précision choisie")
    plt.ylabel("Nombre d'itération")
    plt.title("Nombre d'itération selon la précision en log, question "+str(k))
    plt.legend()
    plt.show()
    
def RayonSpectral(A):
    V=tri_fusion(np.abs(np.linalg.eig(A)[0]))
    n=len(V)
    return V[n-1]

def ChoixOmegaRelaxation(A):   
    D,E,F=Decompose(A)
    x=np.linspace(0,2,10000,endpoint=False)[1:]
    y=[]
    for i in x:
        M=(1/i)*D-E
        if np.linalg.det(M)==0:
            y.append(0)
        else :
            N=((1/i)-1)*D+F
            B=np.dot(np.linalg.inv(M),N)
            k=RayonSpectral(B)
            if k<1:
                y.append(1)
            else:
                y.append(0)
    j=0
    R=[]
    while j<len(x):
        r=[]
        if y[j]==1:
            while y[j]==1:
                r.append(x[j])
                j=j+1
                if j>=len(x):
                    R.append([r[0],r[len(r)-1]])
                    return R
            R.append([r[0],r[len(r)-1]])    
        else :
            j=j+1
    return R        

def MeilleurOmegaRelaxation(A): #fonction qui donne le meilleur omega pour résoudre un système par relaxation
    D,E,F=Decompose(A)
    x=np.linspace(0,2,10000,endpoint=False)[1:]
    y=[]
    for i in x:
        M=(1/i)*D-E
        N=((1/i)-1)*D+F
        B=np.dot(np.linalg.inv(M),N)
        k=RayonSpectral(B)
        y.append(k)
    Xmin=x[0]
    Ymin=y[0]
    i=0
    while i<len (y):
        p=Ymin-y[i]
        if p>0:
            Ymin=y[i]
            Xmin=x[i]
        i=i+1
    return Ymin,Xmin    #Ymin sera le plus petit rayon spectrale de B, atteint pour omega=Xmin
        

def VitesseConvergence(Methode,A,omega=None):
    D,E,F=Decompose(A)
    if (Methode=="Jacobi" or Methode=="J"):
        M=D
        N=E+F
    if (Methode=="GS" or Methode=="Gauss-Seidel"):
        M=D-E
        N=F
    if (Methode=="Relaxation" or Methode=="R"):
        if omega==None:
            omega=MeilleurOmegaRelaxation(A)[1]
        M=(1/omega)*D-E
        N=((1/omega)-1)*D+F
    B=np.dot(np.linalg.inv(M),N)
    return RayonSpectral(B)

def ComparaisonMethode(k,n):
    if k==1:
        A=A1(n)
    elif k==2:
        A=A2(n)
    else :
        return False
    VJ=VitesseConvergence("J",A,omega=None)
    VGS=VitesseConvergence("GS",A,omega=None)
    VR=VitesseConvergence("R",A,omega=None)
    print ("Matrice de la question "+str(k))
    print("Rayon spectral par Jacobi =",VJ)
    print("Rayon spectral par Gauss-Seidel =",VGS)
    print("Rayon spectral par SOR =", VR)
    
    

def ReductionGauss(Aaug):
    
    n,m=Aaug.shape
    
    for k in range (n-1):
        pivot=Aaug[k,k]
        
        if (pivot==0):
            print('le pivot est nul')
            
        elif (pivot!=0):
            
            for i in range(k+1,n):
                Aaug[i,:]=Aaug[i,:]-(Aaug[i,k]/pivot)*Aaug[k,:]

    return(Aaug)


def ResolutionSystTriSup(Taug):
    #Taug=ReductionGauss(Taug)
    n,m=Taug.shape
    X=[0]*n
    #X[n-1]=Taug[n-1][n]/Taug[n-1][n-1]

    for i in range(n-1,-1,-1):
        X[i]=Taug[i][n]

        for j in range(i+1,n):
            X[i]=X[i]-Taug[i][j]*X[j]
        X[i]=X[i]/Taug[i][i]
    return(X)    

def Gauss(A,B):
    n,m=A.shape
    for i in range (n):
        C=np.append(A,B,axis=1)
        
    C=ReductionGauss(C)
    
    C=ResolutionSystTriSup(C)
    
    return (C)

def ComparaisonTempsExecution(k):                               
    TGauss=[]
    TGrSc=[]
    TGaSe=[]
    #TSOR=[]
    N=[]
    for n in range (1,121):
        B=b(n)
        if k==1:
            A=A1(n)
        elif k==2:
            A=A2(n)
        else :
            return False
        t0Gauss=time.time()
        Gauss(A,B)
        tfGauss=time.time()
        TGauss.append(tfGauss-t0Gauss)
        t0GrSc=time.time()
        resolGS(A,B)
        tfGrSc=time.time()
        TGrSc.append(tfGrSc-t0GrSc)
        t0GaSe=time.time()
        MIGaussSeidel(A,B,np.zeros((n,1)),10**-13,500)
        tfGaSe=time.time()
        TGaSe.append(tfGaSe-t0GaSe)
        #t0SOR=time.time()
        #MIRelaxation(A,B,np.zeros((n,1)),10**-13,500,MeilleurOmegaRelaxation(A)[1p])
        #tfSOR=time.time()
        #TSOR.append(tfSOR-t0SOR)
        N.append(n)
    plt.plot(N,TGauss,label="Gauss")
    plt.plot(N,TGrSc,label="Gram-Schmidt")
    plt.plot(N,TGaSe,label="Gauss-Seidel")
    #plt.plot(N,TSOR,label="SOR")
    plt.ylabel("Temps d'éxecution ")
    plt.xlabel("taille du système")
    plt.title("temps d'éxecution en fonction de la taille du système")
    plt.legend()
    plt.show()
    
def pos_def(M):
    A=np.dot(np.transpose(M),M)
    return A


def DecompositionGS(A):                                     
    if np.linalg.det(A)==0:     
        return False                                        
    n=np.shape(A)[0]
    R=np.zeros((n,n))                                       
    Q=np.zeros((n,n))                                       
    R[0,0]=np.linalg.norm(A[:,0])                           
    Q[:,0]= A[:,0]*(1/R[0,0])                               
    for j in range (1,n):
        for i in range (0,j):                               
            R[i,j]=np.vdot(A[:,j],Q[:,i])  
        S=0
        for k in range (j):                                 
            S=S+R[k,j]*Q[:,k]
        w=A[:,j]-S
        R[j,j]=np.linalg.norm(w)                            
        Q[:,j]=w*(1/R[j,j])
    return [Q,R]                    
        
def resolGS(A, b):
    E=DecompositionGS(A)
    Q=E[0]                                                  
    R=E[1]                                                                              
    QT=np.transpose(Q)                                      
    F=np.dot(QT,b)                                          
    X=SysTriSup(R,F)                                        
    return X       
    
def SysTriSup(A,b):                                        
    n= len(b)                                           
    x=np.zeros((n,1))
    x[n-1]=b[n-1]/A[n-1,n-1]
    k=n-1
    while k>-1:
        S=0
        for j in range (k+1,n):
            S=S+A[k,j]*x[j]
        x[k]=(b[k]-S)/A[k,k]
        k=k-1
    return x    
        
        
def MeilleurOmegaRichardson(A):
    n=np.shape(A)[0]
    I=np.eye(n)
    x=np.linspace(0,2,10000,endpoint=False)[1:]
    y=[]
    for i in x: 
        M=I*1/i
        N=-A+I*1/i
        B=np.dot(np.linalg.inv(M),N)
        y.append(RayonSpectral(B))
    Xmin=x[0]
    Ymin=y[0]
    i=0
    while i<len (y):
        p=Ymin-y[i]
        if p>0:
            Ymin=y[i]
            Xmin=x[i]
        i=i+1
    return Ymin,Xmin
    
def MeilleurOmegaJOR(A):
    D=Decompose(A)[0]
    x=np.linspace(0,2,10000,endpoint=False)[1:]           
    y=[]       
    for i in x:
        M=D*1/i
        N=-A+D*1/i
        B=np.dot(np.linalg.inv(M),N)
        y.append(RayonSpectral(B))
    Xmin=x[0]
    Ymin=y[0]
    i=0
    while i<len (y):
        p=Ymin-y[i]
        if p>0:
            Ymin=y[i]
            Xmin=x[i]
        i=i+1
    return Ymin,Xmin    


        
def ComparaisonTempsMoyenExecution():            #des commentaires ont été mis pour mettre en évidence certaine courbe. Par exemple, On n'observe ici que la coube du temps                    
    N=[]                                         #en fonction de la taille du système pour la méthode de Gauss-Seidel. Pour faire apparaitre les autre courbes, il faut 
    #TMGauss=[]                                  #enlever les # devant les variables en "Gauss" pour la méthode directe de Gauss, "GrSc" pour la méthode de Gram-Schmidt,
    #TMGrSc=[]                                   #et "SOR" pour la méthode de relaxation.
    TMGaSe=[]
    #TMSOR=[]
    for n in range (1,121):
        TGauss=[]
        TGrSc=[]
        TGaSe=[]
        #TSOR=[]
        for i in range (0,21):
            A,B=gensys(n)
            #t0Gauss=time.time()
            #Gauss(A,B)
            #tfGauss=time.time()
            #TGauss.append(tfGauss-t0Gauss)
            #t0GrSc=time.time()
            #resolGS(A,B)
            #tfGrSc=time.time()
            #TGrSc.append(tfGrSc-t0GrSc)
            t0GaSe=time.time()
            MIGaussSeidel(A,B,np.zeros((n,1)),10**-13,500)
            tfGaSe=time.time()
            TGaSe.append(tfGaSe-t0GaSe)
            #t0SOR=time.time()
            #MIRelaxation(A,B,np.zeros((n,1)),10**-13,500,MeilleurOmegaRelaxation(A)[1p])
            #tfSOR=time.time()
            #TSOR.append(tfSOR-t0SOR)
        #SGauss=0
        #SGrSc=0
        SGaSe=0
        #SSOR=0
        for i in range (0,21):
            #SGauss+=TGauss[i]
            #SGrSc+=TGrSc[i]
            SGaSe=+TGaSe[i]
            #SSOR=TSOR[i]
        #TMGauss.append(SGauss/21)
        #TMGrSc.append(SGrSc/21)
        TMGaSe.append(SGaSe/21)
        #TMSOR.append(SSOR/21)
        N.append(n)
    #plt.plot(N,TMGauss,label="Gauss")
    #plt.plot(N,TMGrSc,label="Gram-Schmidt")
    plt.plot(N,TMGaSe,label="Gauss-Seidel")
    #plt.plot(N,TMSOR,label="SOR")
    plt.ylabel("Temps d'éxecution ")
    plt.xlabel("taille du système")
    plt.title("temps d'éxecution moyen en fonction de la taille du système")
    plt.legend()
    plt.show()           


           
           
           
           
           
            
