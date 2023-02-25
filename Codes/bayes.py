import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set({})
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: {} is not in my Vocabulary!'.format(word))
    return returnVec


def bagOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # calculate the probabilities of abusive input
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # 防止出现0的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # denominator分母
    p0Denom = sum(p0Num)
    p1Denom = sum(p1Num)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    # 防止下溢出，很多极小的值相乘
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmatian']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split('\\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(
            open('Data/email/spam/{}.txt'.format(i), encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(
            open('Data/email/ham/{}.txt'.format(i), encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 为什么不直接shuffle然后选择前十条或者直接从0-50抽不重复的数据
    # from random import shuffle
    # shuffle(docList)
    # testSet = docList[10]
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainingMat = []
    trainClasses = []
    for docIdx in trainingSet:
        trainingMat.append(bagOfWord2Vec(vocabList, docList[docIdx]))
        trainClasses.append(classList[docIdx])
    p0V, p1V, pSpam = trainNB0(
        np.array(trainingMat), np.array(trainClasses))
    errorCount = 0
    for docIdx in testSet:
        wordVec = bagOfWord2Vec(vocabList, docList[docIdx])
        if classifyNB(np.array(wordVec), p0V, p1V, pSpam) != classList[docIdx]:
            errorCount += 1
            print('classification error', docList[docIdx])
    print('the error rate is: ', float(errorCount/len(testSet)))
