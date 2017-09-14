import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec

def createVacabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# train the classifier with trainMatrix and its trainCategory
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = np.sum(trainCategory)/float(numTrainDocs)

	# p0Num = np.zeros(numWords) ; p1Num = np.ones(numWords)
	# p0Denom = 0.0 ; p1Denom = 0.0

	p0Num = np.ones(numWords) ; p1Num = np.ones(numWords) # init the probablities of p0 and p1
	p0Denom = 2.0 ; p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	
	p0Vect = p0Num/p0Denom
	p1Vect = p1Num/p1Denom
	# p0Vect = np.log(p0Num/p0Denom)
	# p1Vect = np.log(p1Num/p1Denom)
	
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
	p1 = np.sum(vec2Classify * p1Vect) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vect) + np.log(1.0 - pClass1)

	print(p0, ' ', p1)

	if p1 > p0: return 1
	else: return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()

	vocabList = createVacabList(listOPosts)
	trainMatrix = []

	for postinDoc in listOPosts:
		trainMatrix.append(setOfWords2Vec(vocabList, postinDoc))
	
	p0V, p1V, pAb = trainNB0(trainMatrix, listClasses)
	print(p0V, p1V, pAb)

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

	testEntry = ['stupid', 'garbage']
	thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))		

def main():
	testingNB()

if __name__ == '__main__':
	main()