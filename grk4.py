#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd

from matplotlib import cm

para=[9.81,.4,.4,1,1]

A2=np.load('dataqp-RK4_h3.npy')

def energias(M):
    g,l1,l2,m1,m2=para
    n=len(M)
    E=np.zeros((n,4),dtype=np.float64)
    for i in range(n):
        E[i,0]=M[i,0]
        E[i,1]=-(m1+m2)*g*l1*np.cos(M[i,1])-m2*g*l2*np.cos(M[i,2])
        E[i,2]=(m2*l2**2*M[i,3]**2+(m1+m2)*l1**2*M[i,4]**2-2*m2*l1*l2*M[i,3]*M[i,4]*np.cos(M[i,1]-M[i,2]))/(2*m2*l1**2*l2**2*(m1+m2*np.sin(M[i,1]-M[i,2])**2))
        #E[i,2]=.5*m1*l1**2*M[i,3]**2+.5*m2*(l1**2*M[i,3]**2+l2**2*M[i,4]**2+2*l1*l2*M[i,3]*M[i,4]*np.cos(M[i,1]-M[i,2]))
        E[i,3]=E[i,1]+E[i,2]
    return E

E2=energias(A2)

# Grafica energias vs tiempo
#plt.figure(1)
fig,axx=plt.subplots(2,sharex=True)
axx[0].plot(E2[:,0],E2[:,1],label='$E_p$',lw=1,color='C1')
axx[0].plot(E2[:,0],E2[:,2],label='$E_k$',lw=1,color='C2')
axx[0].plot(E2[:,0],E2[:,3],label='$E_T$',lw=1,color='C0')
#plt.title('Evolución temporal de Energías')
axx[1].plot(E2[:,0],E2[:,3],label='$E_T$',lw=1,color='C0')
axx[1].set_xlabel('$t \;[s]$')
axx[0].set_ylabel('$E \;[J]$')
axx[1].set_ylabel('$E \;[J]$')
axx[1].legend()
axx[0].legend()

####################################################
# Grafica espacio de fases
Pmo=np.zeros((len(A2),2),dtype=np.float64)
g,l1,l2,m1,m2=para
for i in range(len(A2)):
    Pmo[i,0]=(m1+m2)*l1**2*A2[i,3]+m2*l1*l2*A2[i,4]*np.cos(A2[i,1]-A2[i,2])
    Pmo[i,1]=m2*l2**2*A2[i,4]+m2*l1*l2*A2[i,3]*np.cos(A2[i,1]-A2[i,2])

plt.figure(3)
#plt.plot(A2[:,1],Pmo[:,0],label=r'$p_1\;vs\;q_1$')
#plt.plot(A2[:,2],Pmo[:,1],label=r'$p_2\;vs\;q_2$')
plt.plot(A2[:,1],A2[:,3],label=r'$p_1\;vs\;q_1$')
plt.plot(A2[:,2],A2[:,4],label=r'$p_2\;vs\;q_2$')
#plt.title('Espacio de fases')
plt.xlabel(r'$q_i$')
plt.ylabel(r'$p_i$')
plt.legend()

plt.show()

