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


def H_eulerp(v,h,tf):
    t0,q1,q2,p1,p2=v # valores _(n)
    g,l1,l2,m1,m2=para
    vx=array(v)
    step=int((tf-t0)/h)
    M=zeros((step+1,len(v)),dtype=float64) # valores _(n+1)
    M[0]=vx
    J=zeros((step+1,1),dtype=float64)
    q1,q2,p1,p2=sy.symbols('q1 q2 p1 p2')
    sis=simbol(h)
    num=sy.lambdify([q1,q2,p1,p2],sis)
    for i in range(1,step+1):
        t0,q1,q2,p1,p2=M[i-1]
        A=h*sin(q1-q2)*(m2*sin(q1-q2)**2+2*m2*cos(q1-q2)**2+m1)/((m1+m2*sin(q1-q2)**2))**2/l1/l2
        C=-h*(m1+m2)*g*l1*sin(q1)+p1
        D=-h*m2*g*l2*sin(q2)+p2
        qq=2*m2*l1**2*l2**2*(m1+m2*sin(q1-q2)**2)
        M[i,0]=i*h
        a=num(q1,q2,p1,p2)
        M[i,3]=roots([a[0][0],a[0][1],a[0][2]])[1]
        M[i,4]=roots([a[1][0],a[1][1],a[1][2]])[1]
        M[i,1]=q1+h*(2*m2*l2**2*M[i,3]-2*m2*l1*l2*M[i,4]*cos(q1-q2))/qq
        M[i,2]=q2+h*(2*(m1+m2)*l1**2*M[i,4]-2*m2*l1*l2*M[i,3]*cos(q1-q2))/qq
    return M

AA=H_eulerp([0,pi/2,0*pi/2,0.,0.],1e-5,20)
save('data-ES_h5',AA)



