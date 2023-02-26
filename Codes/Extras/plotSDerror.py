'''
创建时间：20230226
创建人：Stephen CUI
'''

import logRegres
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = .01
    weights = ones(n)
    weightHistory = zeros((200 * m, n))
    for j in range(200):
        for i in range(m):
            h = logRegres.sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightHistory[j * m + i, :] = weights
    return weightHistory


def stocGradAscent1(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    weights = ones(n)
    weightHistory = zeros((40 * m, n))
    for j in range(40):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + j + i) + .01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = logRegres.sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            weightHistory[j * m + i, :] = weights
            del dataIndex[randIndex]
    return weightHistory


dataMat, labelMat = logRegres.loadDataSet()
dataArr = array(dataMat)
# myHist = stocGradAscent0(dataArr, labelMat)
myHist = stocGradAscent1(dataArr, labelMat)

n = shape(dataArr)[0]
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []

fig = plt.figure()
ax = fig.add_subplot(311)
type1 = ax.plot(myHist[:, 0])
plt.ylabel('X0')
ax = fig.add_subplot(312)
type1 = ax.plot(myHist[:, 1])
plt.ylabel('X1')
ax = fig.add_subplot(313)
type1 = ax.plot(myHist[:, 2])
plt.xlabel('iteration')
plt.ylabel('X2')
plt.show()
