#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:12:02 2024

@author: glennwagner
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import rc
rc('font',**{'size'   : 15,'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def tanh(M):
    E,U=np.linalg.eig(M)
    return np.einsum("ab,bc,cd->ad",U,np.diag(np.tanh(E)),np.linalg.inv(U),optimize=True)
 

eC=1.6*10**-19
hbar=1.05*10**-34



N=20 # cutoff


vbar=2.60*eC*10**-10
m=0.055*eC*2
delta=-0.11*eC
Delta0=0.011*eC*2

Delta=Delta0
epsilon=Delta0*1


S0 = np.eye(2)
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

Tau0 = np.eye(2)
Taux = np.array([[0, 1], [1, 0]])
Tauy = np.array([[0, -1j], [1j, 0]])
Tauz = np.array([[1, 0], [0, -1]])


A=np.zeros((N,N)) 
for n in range(N):
    A[n-1,n]=np.sqrt(n) 
Ad=A.T.conj()
A_sq=np.matmul(Ad,A)
I=np.eye(N)

kylims=50
xlims=100
Elims=70

xarr=np.linspace(-xlims,xlims,100)
Earr=np.linspace(-Elims,Elims,100)
XX,EE=np.meshgrid(xarr,Earr,indexing="ij")
LDOS=np.zeros_like(XX)

fig, axs = plt.subplots(2,1,figsize=(6,5))   

for l in range(2):

    Delta=Delta0*l
    epsilon=Delta0*(1-l)

    xlist=np.arange(-20,20,0.5)
    
    B=12
    Elist=np.zeros((len(xlist),4*N))
    xexp=np.zeros((len(xlist),4*N))
    occ=np.zeros((len(xlist),4*N))
    
    for idx,x in enumerate(xlist):
        ellB=np.sqrt(hbar/(eC*B))
        
        X=ellB*(A+Ad)/np.sqrt(2)
        KX=-1j*(A-Ad)/np.sqrt(2)/ellB
    
        Ham=np.zeros((4*N,4*N),dtype=complex)
        
        Ham+=vbar*np.kron(np.kron(Sy,Tau0),KX)
        Ham+=vbar*np.kron(np.kron(Sx,Tau0),X/ellB**2)
    
        xarg=X/ellB+x*I
        marg=-I*tanh(xarg)
        
        Ham+=m*np.kron(np.kron(S0,Taux),marg)
        Ham+=delta*np.kron(np.kron(Sx,Tauy),marg)
        Ham+=epsilon*np.kron(np.kron(S0,Tauz),I)*0
        Ham+=Delta*np.kron(np.kron(Sz,Tauy),marg)
        
        
        evals,evecs=np.linalg.eigh(Ham.reshape((4*N,4*N)))
        
        Elist[idx]=evals
    
        Y4=np.kron(np.kron(S0,Tau0),X)
        
        xexp[idx]=np.einsum("ab,ac,cb->b",evecs.conj(),Y4,evecs,optimize=True).real+x*ellB
        
        Nop=np.kron(np.kron(S0,Tau0),A_sq)
        occ[idx]=np.einsum("ab,ac,cb->b",evecs.conj(),Nop,evecs,optimize=True).real
        
    eps_eV=epsilon/eC    
    Del_eV=Delta/eC    
        
    
    if Delta/eC<0.0001 and epsilon/eC<0.0001:
        color="b"
    if Delta/eC<0.0001 and epsilon/eC>0.0001:
        color="cyan"
    if Delta/eC>0.0001 and epsilon/eC<0.0001:
        color="r"
    
    
    
    klist=xlist[:,None]*np.ones(4*N)[None,:]


    delta_E=2
    delta_x=5

    
    states_x=xexp[occ<0.5*N].flatten()*10**9
    states_E=Elist[occ<0.5*N].flatten()*1000/eC
    for i in range(len(states_x)):
        LDOS+=np.exp(-(EE-states_E[i])**2/delta_E**2)*np.exp(-(XX-states_x[i])**2/delta_x**2) 
    
    cmapx=axs[1].scatter(Elist[occ<0.5*N]*1000/eC,klist[occ<0.5*N],c=xexp[occ<0.5*N]*10**9,cmap="seismic",vmin=-70,vmax=70,marker="o")
    axs[1].set_xlim([-40,40])
    axs[1].set_ylim([-7,7])
    axs[1].set_ylabel("$k_y\ell$")
    axs[1].set_xlabel(r"$E \ \textrm{(meV)}$")

    cmap=axs[0].pcolor(EE,XX,LDOS,cmap="jet",vmax=5)
    axs[0].set_ylim([-Elims,Elims])
    axs[0].set_ylabel(r"$x \ \textrm{(nm)}$")
    axs[0].set_xlabel(r"$E \ \textrm{(meV)}$")
    
clb=fig.colorbar(cmapx,ax=axs[1])
clb.ax.set_title(r"$\langle x\rangle \ \textrm{(nm)}$",fontsize=10)

clb=fig.colorbar(cmap,ax=axs[0])
clb.ax.set_title(r"$\textrm{DOS}$",fontsize=10)

plt.tight_layout()
plt.savefig("./figures/LL_spectrum_delta_positive.pdf",bbox_inches='tight',pad_inches = 0)
plt.show()




