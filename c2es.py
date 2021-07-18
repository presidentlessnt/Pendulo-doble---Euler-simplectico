#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sympy as sy
import matplotlib.ticker as mtick

para=[9.81,.4,.4,1,1]

def Hamil(v):
    t0,q1,q2,p1,p2=v
    g,l1,l2,m1,m2=para
    T=(m2*l2**2*p1**2+(m1+m2)*l1**2*p2**2-2*m2*l1*l2*p1*p2*cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sin(q1-q2)**2))
    V=-(m1+m2)*g*l1*cos(q1)-m2*g*l2*cos(q2)
    return t0,V,T,T+V


def simbol(h):
    q1,q2,p1,p2=sy.symbols('q1 q2 p1 p2')
    g,l1,l2,m1,m2=para
    x,y=sy.symbols('x y')
    P1Ha=(m2*l2**2*x**2+(m1+m2)*l1**2*p2**2-2*m2*l1*l2*x*p2*sy.cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sy.sin(q1-q2)**2))-(m1+m2)*g*l1*sy.cos(q1)-m2*g*l2*sy.cos(q2)
    mP1Ha=sy.Matrix([P1Ha])
    vP1=sy.Matrix([q1])
    jP1=mP1Ha.jacobian(vP1)[0]*(-h)+p1-x
    coeP1=sy.Poly(jP1,x).all_coeffs()
    P2Ha=(m2*l2**2*p1**2+(m1+m2)*l1**2*y**2-2*m2*l1*l2*p1*y*sy.cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sy.sin(q1-q2)**2))-(m1+m2)*g*l1*sy.cos(q1)-m2*g*l2*sy.cos(q2)
    mP2Ha=sy.Matrix([P2Ha])
    vP2=sy.Matrix([q2])
    jP2=mP2Ha.jacobian(vP2)[0]*(-h)+p2-y
    coeP2=sy.Poly(jP2,y).all_coeffs()
    return coeP1,coeP2


def jaco(sis,h):
    q1,q2,p1,p2=sy.symbols('q1 q2 p1 p2')
    g,l1,l2,m1,m2=para
    coeP1,coeP2=sis
    PP1=(-coeP1[1]-(coeP1[1]**2-4*coeP1[0]*coeP1[2])**.5)/2/coeP1[0]
    PP2=(-coeP2[1]-(coeP2[1]**2-4*coeP2[0]*coeP2[2])**.5)/2/coeP2[0]
    QQ2=q2+h*(2*l1**2*PP2*(m1+m2)-2*l1*l2*m2*PP1*sy.cos(q1-q2))/(2*l1**2*l2**2*m2*(m1+m2*sy.sin(q1-q2)**2))
    QQ1=q1+h*(-2*l1*l2*m2*PP2*sy.cos(q1-q2)+2*l2**2*m2*PP1)/(2*l1**2*l2**2*m2*(m1+m2*sy.sin(q1-q2)**2))
    HH=sy.Matrix([QQ1,QQ2,PP1,PP2])
    V=sy.Matrix([q1,q2,p1,p2])
    return HH.jacobian(V)


def factor_vol(A,h):
    t0,q1,q2,p1,p2=A[0]
    g,l1,l2,m1,m2=para
    step=int((A[-1][0]-A[0][0])/h)
    J=zeros((step,2),dtype=float64)
    q1,q2,p1,p2=sy.symbols('q1 q2 p1 p2')
    sis=simbol(h)
    jcb=jaco(sis,h)
    den=sy.lambdify([q1,q2,p1,p2],jcb)
    for i in range(step):
        t0,q1,q2,p1,p2=A[i]
        b=den(q1,q2,p1,p2)
        J[i,0]=i*h
        J[i,1]=linalg.det(den(q1,q2,p1,p2))
    return J


def factor_volX(A,h,X):
    t0,q1,q2,p1,p2=A[0]
    g,l1,l2,m1,m2=para
    step=len(A)
    J=zeros((step//X+1,2),dtype=float64)
    q1,q2,p1,p2=sy.symbols('q1 q2 p1 p2')
    sis=simbol(h)
    jcb=jaco(sis,h)
    den=sy.lambdify([q1,q2,p1,p2],jcb)
    cont=0
    for i in range(step):
        if i%X==0:
            t0,q1,q2,p1,p2=A[i]
            J[cont,0]=i*h
            J[cont,1]=linalg.det(den(q1,q2,p1,p2))
            cont+=1
    return J

PP=load('data-ES_h5.npy')

AA=factor_volX(PP,1e-5,97)
save('det-ES_h5',AA)


