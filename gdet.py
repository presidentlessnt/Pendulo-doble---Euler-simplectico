#! /usr/bin/python3
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sympy as sy
import matplotlib.ticker as mtick

para=[9.81,.4,.4,1,1]

GA=load('det-ES_h2.npy')
GE=load('det-RK4_h2.npy')


fig,axx=plt.subplots(2,sharex=True)
plt.suptitle('h=$10^{-2}$ [s]')
axx[0].plot(GA[:,0],GA[:,1],label='Symplectic-Euler')
axx[1].plot(GE[:,0],GE[:,1],color='C1',label='RK4')
axx[1].ticklabel_format(useOffset=True, style='scientific', axis='y')
#axx[0].ticklabel_format(useOffset=False)
#axx[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.7f'))
axx[0].set_ylabel('$\Gamma_{vol}$')
axx[1].set_ylabel('$\Gamma_{vol}$')
axx[1].set_xlabel('t [s]')
axx[0].legend(loc=4)
axx[1].legend(loc=4)

plt.show()

