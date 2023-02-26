'''
创建时间：20230226
创建人：Stephen CUI
'''

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

leafNode = dict(boxstyle='round4', fc='.8')
arrow_args = dict(arrowstyle='<-')

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = .025
x = np.arange(-2, 2, delta)
y = np.arange(-2, 2, delta)
X, Y = np.meshgrid(x, y)
Z1 = -((X - 1) ** 2)
Z2 = -(Y ** 2)
Z = (Z2 + Z1) + 5.0

plt.figure()
CS = plt.contour(X, Y, Z)
plt.annotate('', xy=(.05, .05), xycoords='axes fraction', xytext=(.2, .2),
             textcoords='axes fraction', va='center', ha='center', bbox=leafNode, arrowprops=arrow_args)
plt.text(-1.9, -1.8, 'P0')
plt.annotate('', xy=(0.2, 0.2),  xycoords='axes fraction',
             xytext=(0.35, 0.3), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args)
plt.text(-1.35, -1.23, 'P1')
plt.annotate('', xy=(0.35, 0.3),  xycoords='axes fraction',
             xytext=(0.45, 0.35), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args)
plt.text(-0.7, -0.8, 'P2')
plt.text(-0.3, -0.6, 'P3')
plt.clabel(CS, inline=1, fontsize=10)
# plt.title('Gradient Ascent')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
