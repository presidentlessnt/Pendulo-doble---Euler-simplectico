#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd
import sympy as sy


para=[9.81,.4,.4,1,1]

#q1,q2,p1,p2,g,l1,l2,m1,m2,h=sy.symbols('q1 q2 p1 p2 g l1 l2 m1 m2 h') # puro simbolico
q1,q2,p1,p2=sy.symbols('q1 q2 p1 p2')
g,l1,l2,m1,m2=para
h=1e-4

Ha=(m2*l2**2*p1**2+(m1+m2)*l1**2*p2**2-2*m2*l1*l2*p1*p2*sy.cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sy.sin(q1-q2)**2))-(m1+m2)*g*l1*sy.cos(q1)-m2*g*l2*sy.cos(q2)

H=sy.Matrix([Ha])
V=sy.Matrix([q1,q2,p1,p2])

Jcb=H.jacobian(V)
#print(Jcb[0],'\n\n',Jcb[1],'\n\n',Jcb[2],'\n\n',Jcb[3],'\n\n')
#print(sy.simplify(Jcb[0,0]))
#print(Jcb.det())


## CASO P1_(n+1) OR P2_(n+1) POR SEPARADOS
x,y=sy.symbols('x y')

P1Ha=(m2*l2**2*x**2+(m1+m2)*l1**2*p2**2-2*m2*l1*l2*x*p2*sy.cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sy.sin(q1-q2)**2))-(m1+m2)*g*l1*sy.cos(q1)-m2*g*l2*sy.cos(q2)
mP1Ha=sy.Matrix([P1Ha])
vP1=sy.Matrix([q1])
jP1=mP1Ha.jacobian(vP1)[0]*(-h)+p1-x
coeP1=sy.Poly(jP1,x).all_coeffs()
print(coeP1,'\n')

P2Ha=(m2*l2**2*p1**2+(m1+m2)*l1**2*y**2-2*m2*l1*l2*p1*y*sy.cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sy.sin(q1-q2)**2))-(m1+m2)*g*l1*sy.cos(q1)-m2*g*l2*sy.cos(q2)
mP2Ha=sy.Matrix([P2Ha])
vP2=sy.Matrix([q2])
jP2=mP2Ha.jacobian(vP2)[0]*(-h)+p2-y
coeP2=sy.Poly(jP2,y).all_coeffs()
print(coeP2,'\n')

#print(sy.solve(coeP1[0]*x**2+coeP1[1]*x+coeP1[2],x))
#print((-coeP1[1]+(coeP1[1]**2-4*coeP1[0]*coeP1[2])**.5)/2/coeP1[0])
PP1=(-coeP1[1]-(coeP1[1]**2-4*coeP1[0]*coeP1[2])**.5)/2/coeP1[0]
PP2=(-coeP2[1]-(coeP2[1]**2-4*coeP2[0]*coeP2[2])**.5)/2/coeP2[0]


QQ2=q2+h*(2*l1**2*PP2*(m1+m2)-2*l1*l2*m2*PP1*sy.cos(q1-q2))/(2*l1**2*l2**2*m2*(m1+m2*sy.sin(q1-q2)**2))
QQ1=q1+h*(-2*l1*l2*m2*PP2*sy.cos(q1-q2)+2*l2**2*m2*PP1)/(2*l1**2*l2**2*m2*(m1+m2*sy.sin(q1-q2)**2))

HH=sy.Matrix([QQ1,QQ2,PP1,PP2])

#asd=HH.jacobian(V)
#print(type(asd))

#for i in range(4):
#    print(asd[i],'\n')

#dsa=asd.subs([(q1,1.53),(q2,0),(p1,0),(p2,0)]).evalf()
#print(dsa)
#print(dsa.det())

