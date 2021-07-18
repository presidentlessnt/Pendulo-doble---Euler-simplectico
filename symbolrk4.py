#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd
import sympy as sy


para=[9.81,.4,.4,1,1]

#q1,q2,p1,p2,g,l1,l2,m1,m2,h=sy.symbols('q1 q2 p1 p2 g l1 l2 m1 m2 h') # puro simbolico
q1,q2,p1,p2,h=sy.symbols('q1 q2 p1 p2 h')
g,l1,l2,m1,m2=para
#h=1e-3

Ha=(m2*l2**2*p1**2+(m1+m2)*l1**2*p2**2-2*m2*l1*l2*p1*p2*sy.cos(q1-q2))/(2*m2*l1**2*l2**2*(m1+m2*sy.sin(q1-q2)**2))-(m1+m2)*g*l1*sy.cos(q1)-m2*g*l2*sy.cos(q2)


def pp1(q1,q2,p1,p2):
    return g*l1*(m1 + m2)*sy.sin(q1) + p1*p2*sy.sin(q1 - q2)/(l1*l2*(m1 + m2*sy.sin(q1 - q2)**2)) - (l1**2*p2**2*(m1 + m2) - 2*l1*l2*m2*p1*p2*sy.cos(q1 - q2) + l2**2*m2*p1**2)*sy.sin(q1 - q2)*sy.cos(q1 - q2)/(l1**2*l2**2*(m1 + m2*sy.sin(q1 - q2)**2)**2) 

def pp2(q1,q2,p1,p2):
    return g*l2*m2*sy.sin(q2) - p1*p2*sy.sin(q1 - q2)/(l1*l2*(m1 + m2*sy.sin(q1 - q2)**2)) + (l1**2*p2**2*(m1 + m2) - 2*l1*l2*m2*p1*p2*sy.cos(q1 - q2) + l2**2*m2*p1**2)*sy.sin(q1 - q2)*sy.cos(q1 - q2)/(l1**2*l2**2*(m1 + m2*sy.sin(q1 - q2)**2)**2) 

def ww1(q1,q2,p1,p2):
    return (-2*l1*l2*m2*p2*sy.cos(q1 - q2) + 2*l2**2*m2*p1)/(2*l1**2*l2**2*m2*(m1 + m2*sy.sin(q1 - q2)**2))

def ww2(q1,q2,p1,p2):
    return (2*l1**2*p2*(m1 + m2) - 2*l1*l2*m2*p1*sy.cos(q1 - q2))/(2*l1**2*l2**2*m2*(m1 + m2*sy.sin(q1 - q2)**2))


def F(q1,q2,p1,p2):
    w1=ww1(q1,q2,p1,p2)
    w2=ww2(q1,q2,p1,p2)
    return -(m2*w1**2*sy.sin(2*q1-2*q2)+2*m2*l2/l1*w2**2*sy.sin(q1-q2)+2*g/l1*((m1+m2)*sy.sin(q1)-m2*sy.sin(q2)*sy.cos(q1-q2)))/(2*m1+m2-m2*sy.cos(2*q1-2*q2))


def G(q1,q2,p1,p2):
    w1=ww1(q1,q2,p1,p2)
    w2=ww2(q1,q2,p1,p2)
    return (2*(m1+m2)*l1/l2*w1**2*sy.sin(q1-q2)+m2*w2**2*sy.sin(2*q1-2*q2)+2*g/l2*(m1+m2)*(sy.sin(q1)*sy.cos(q1-q2)-sy.sin(q2)))/(2*m1+m2-m2*sy.cos(2*q1-2*q2))


#### K_1

k1q1=ww1(q1,q2,p1,p2)

k1q2=ww2(q1,q2,p1,p2)

k1p1=pp1(q1,q2,p1,p2)

k1p2=pp2(q1,q2,p1,p2)

#### K_2

k2q1=ww1(q1+h/2*k1q1,q2+h/2*k1q2,p1+h/2*k1p1,p2+h/2*k1p2)

k2q2=ww2(q1+h/2*k1q1,q2+h/2*k1q2,p1+h/2*k1p1,p2+h/2*k1p2)

k2p1=pp1(q1+h/2*k1q1,q2+h/2*k1q2,p1+h/2*k1p1,p2+h/2*k1p2)

k2p2=pp2(q1+h/2*k1q1,q2+h/2*k1q2,p1+h/2*k1p1,p2+h/2*k1p2)

#### K_3

k3q1=ww1(q1+h/2*k2q1,q2+h/2*k2q2,p1+h/2*k2p1,p2+h/2*k2p2)

k3q2=ww2(q1+h/2*k2q1,q2+h/2*k2q2,p1+h/2*k2p1,p2+h/2*k2p2)

k3p1=pp1(q1+h/2*k2q1,q2+h/2*k2q2,p1+h/2*k2p1,p2+h/2*k2p2)

k3p2=pp2(q1+h/2*k2q1,q2+h/2*k2q2,p1+h/2*k2p1,p2+h/2*k2p2)

#### K_4

k4q1=ww1(q1+h*k3q1,q2+h*k3q2,p1+h*k3p1,p2+h*k3p2)

k4q2=ww2(q1+h*k3q1,q2+h*k3q2,p1+h*k3p1,p2+h*k3p2)

k4p1=pp1(q1+h*k3q1,q2+h*k3q2,p1+h*k3p1,p2+h*k3p2)

k4p2=pp2(q1+h*k3q1,q2+h*k3q2,p1+h*k3p1,p2+h*k3p2)

#### FINAL

Tq1=q1+h/6*(k1q1+2*k2q1+2*k3q1+k4q1)

Tq2=q2+h/6*(k1q2+2*k2q2+2*k3q2+k4q2)

Tp1=p1+h/6*(k1p1+2*k2p1+2*k3p1+k4p1)

Tp2=p2+h/6*(k1p2+2*k2p2+2*k3p2+k4p2)

#print(Tp1)


H=sy.Matrix([Tq1,Tq2,Tp1,Tp2])
V=sy.Matrix([q1,q2,p1,p2])

AA=H.jacobian(V)


variable_list = [q1, q2, p1, p2]

jacobian_lines = ["def jacobian(variables,h):",
                           '    """ Returns the evaluated jacobian matrix',
                           '    :param variables: a list of numeric values to evaluate the jacobian',
                           '    """', '', '    {} = variables'.format(str(variable_list)), '',
                           '    j = {}'.format(AA), '', "    return j"]

file_path = 'nobo-rk4.py'
file = open(file_path, 'w').write('\n'.join(jacobian_lines))
