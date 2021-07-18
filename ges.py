#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sympy as sy

para=[9.81,.4,.4,1,1]

def Hamil(v):
    t0,q1,q2,p1,p2=v
    g,l1,l2,m1,m2=para
    T=(m2*l2**2*p1**2+(m1+m2)*l1**2*p2**2-2*m2*l1*l2*p1*p2*cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sin(q1-q2)**2))
    V=-(m1+m2)*g*l1*cos(q1)-m2*g*l2*cos(q2)
    return t0,V,T,T+V

AA=load('data-ES_h2.npy')

def E_mtx(M):
    E=zeros((len(M),4),dtype=float64)
    for i in range(len(M)):
        E[i]=Hamil(M[i])
    return E

EE=E_mtx(AA)

# Grafica energias vs tiempo
fig,axx=plt.subplots(2,sharex=True)
axx[0].plot(EE[:,0],EE[:,1],label='$E_p$',linewidth=.5,color='C1')
axx[0].plot(EE[:,0],EE[:,2],label='$E_k$',linewidth=.5,color='C2')
axx[0].plot(EE[:,0],EE[:,3],label='$E_T$',linewidth=.5,color='C0')
#plt.title('Evolución temporal de Energías')
axx[1].plot(EE[:,0],EE[:,3],label='$E_T$',linewidth=.5,color='C0')
axx[1].set_xlabel('$t \;[s]$')
axx[0].set_ylabel('$E \;[J]$')
axx[1].set_ylabel('$E \;[J]$')
axx[1].legend()
axx[0].legend()

# Grafica espacio de fases
plt.figure(2)
plt.plot(AA[:,1],AA[:,3],label=r'$p_1\;vs\;q_1$')
plt.plot(AA[:,2],AA[:,4],label=r'$p_2\;vs\;q_2$')
#plt.title('Espacio de configuraciones')
plt.xlabel(r'$q_i$')
plt.ylabel(r'$p_i$')
plt.legend()

plt.show()
