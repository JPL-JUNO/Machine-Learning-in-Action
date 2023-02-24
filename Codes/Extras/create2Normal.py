'''
创建时间：20230224
创建人： Stephen CUI
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

n = 1_000
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers = ['^', 'o']
# fw = open('')
for i in range(n):
    [r0, r1] = np.random.standard_normal(2)
    myClass = np.random.uniform(0, 1)
    if (myClass <= .5):
        fFlyer = r0 + 9.0
        tats = r1 + r0
        xcord0.append(fFlyer)
        ycord0.append(tats)
    else:
        fFlyer = r0 + 2
        tats = r1 + r0
        xcord1.append(fFlyer)
        ycord1.append(tats)
    # fw.write('{}\t {}\n'.format(fFlyer, tats))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0, ycord0, marker=markers[0], s=90)
ax.scatter(xcord1, ycord1, marker=markers[1], s=50, c='red')
plt.show()
