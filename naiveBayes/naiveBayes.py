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

	# p0Num = np.ones(numWords) ; p1Num = np.ones(numWords) # init the probablities of p0 and p1
	# p0Denom = 2.0 ; p1Denom = 2.0

	p0Num = np.zeros(numWords) ; p1Num = np.ones(numWords)
	p0Denom = 0.0 ; p1Denom = 0.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	
	# p0Vect = np.log(p0Num/p0Denom)
	# p1Vect = np.log(p1Num/p1Denom)
	p0Vect = p0Num/p0Denom
	p1Vect = p1Num/p1Denom
	
	return p0Vect, p1Vect, pAbusive

def main():
	listOPosts, listClasses = loadDataSet()
	vocabList = createVacabList(listOPosts)
	
	# vec = setOfWords2Vec(vocabList, ('my', 'dog', 'is', 'cute', 'very', 'cute'))
	trainMatrix = []

	for postinDoc in listOPosts:
		trainMatrix.append(setOfWords2Vec(vocabList, postinDoc))
	
	p0V, p1V, pAb = trainNB0(trainMatrix, listClasses)

	print(pAb)
	print(p0V)
	print(p1V)

if __name__ == '__main__':
	main()