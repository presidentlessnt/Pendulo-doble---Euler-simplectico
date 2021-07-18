#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd
import sympy as sy

a,b,c,d=sy.symbols('a b c d')

eq = [a*b, a + b*c -d, c**2 + a, d*a - 2]

eq=sy.Matrix(eq)

jacobi = eq.jacobian([a, b, c, d])

variable_list = [a, b, c, d]

jacobian_lines = ["def jacobian(variables):",
                           '    """ Returns the evaluated jacobian matrix',
                           '    :param variables: a list of numeric values to evaluate the jacobian',
                           '    """', '', '    {} = variables'.format(str(variable_list)), '',
                           '    j = {}'.format(jacobi), '', "    return j"]

file_path = 'jacobian1.py'
file = open(file_path, 'w').write('\n'.join(jacobian_lines))

'''
jacobian_lines = ['    {} = variables'.format(str(variable_list)),'','    j = {}'.format(jacobi)]

#kk=[jacobi[0],jacobi[1],jacobi[2],jacobi[3],jacobi[4]]#['{}'.format(jacobi[i] for i in range(4))]

file_path = 'jacobian3.py'

file = open(file_path, 'w')

#for line in jacobian_lines:
#    file.write('{}\n'.format(line))

#for line in kk:
#    file.write('{}\n'.format(line))

file.write('\n\n\n{}\n'.format(jacobi.det()))

#file = open(file_path, 'w').write('\n'.join(jacobian_lines))

#print(load('RK4-jaco.npy',allow_pickle=True))
'''