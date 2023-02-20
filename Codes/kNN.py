import numpy as np
import operator
from os import listdir


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, .1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # Construct an array by repeating A the number of times given by reps.
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** .5
    # Returns the indices that would sort an array.
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    #  same with sortedClassCount = sorted(classCount.items(), reverse=True)?
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filenames):
    loveDict = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filenames)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(loveDict.get(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = .1
    datingDataMat, datingLabels = file2matrix('Data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVectors = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVectors):
        classifierRet = classify0(
            normMat[i, :], normMat[numTestVectors:m, :], datingLabels[numTestVectors:m], 3)
        print('the classifier came back with {}, the real answer is {}'.format(
            classifierRet, datingLabels[i]))
        if (classifierRet != datingLabels[i]):
            errorCount += 1
    print('the total error rate is {:.2f}'.format(errorCount / numTestVectors))
    print(errorCount)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('Data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(
        (inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person: {}'.format(
        resultList[classifierResult - 1]))


def img2vector(filename):
    returnMat = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnMat[0, 32*i+j] = int(lineStr[j])
    return returnMat


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('Data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(
            'Data/trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('Data/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('Data/testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('The classifier came back with: {}, the real answer is : {}'.format(
            classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
    print('\nThe total number of errors is: {}'.format(errorCount))
    print('\nThe total error rate is {:.2f}'.format(
        errorCount/float(mTest)))
