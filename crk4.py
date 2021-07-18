#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd

from matplotlib import cm

para=[9.81,.4,.4,1,1]

def mfuno2(v):
    t0,o1,o2,w1,w2=v
    g,l1,l2,m1,m2=para
    A=-(m2*w1**2*np.sin(2*(o1-o2))+2*m2*(l2/l1)*w2**2*np.sin(o1-o2)+2*(g/l1)*((m1+m2)*np.sin(o1)-m2*np.sin(o2)*np.cos(o1-o2)))/((2*m1+m2)-m2*np.cos(2*(o1-o2)))
    B=(2*(m1+m2)*(l1/l2)*w1**2*np.sin(o1-o2)+m2*w2**2*np.sin(2*(o1-o2))+2*(g/l2)*(m1+m2)*(np.sin(o1)*np.cos(o1-o2)-np.sin(o2)))/((2*m1+m2)-m2*np.cos(2*(o1-o2)))
    return np.array([0,w1,w2,A,B])

A2=mtd.medo_rk4(mfuno2,[0,np.pi/2,0*np.pi/2,0.,0.],1e-5,0,20)

Pmo=np.copy(A2)
g,l1,l2,m1,m2=para
for i in range(len(A2)):
    Pmo[i,3]=(m1+m2)*l1**2*A2[i,3]+m2*l1*l2*A2[i,4]*np.cos(A2[i,1]-A2[i,2])
    Pmo[i,4]=m2*l2**2*A2[i,4]+m2*l1*l2*A2[i,3]*np.cos(A2[i,1]-A2[i,2])

np.save('dataqp-RK4_h5',Pmo)
