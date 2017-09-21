import re

import feedparser
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]	#1 is abusive, 0 not
	return postingList, classVec

def createVacabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

# 词集模型
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		# else: print("the word: %s is not in my Vocabulary!" % word)
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
		if trainCategory[i] > 0:
			p1Num += trainMatrix[i]
			p1Denom += np.sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += np.sum(trainMatrix[i])
	
	# p0Vect = p0Num/p0Denom
	# p1Vect = p1Num/p1Denom
	p0Vect = np.log(p0Num/p0Denom)
	p1Vect = np.log(p1Num/p1Denom)
	
	return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
	p1 = np.sum(vec2Classify * p1Vect) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vect) + np.log(1.0 - pClass1)

	# print(p0, ' ', p1)

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

# 词袋模型，某些词出现不止一次可能会表达更多的信息
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			pass
		# else: print("the word: %s is not in my Vocabulary!" % word)
	return returnVec

def parseTxt(bigString):
	pattern = re.compile(r'\w*', re.S)
	words = re.findall(pattern, bigString)
	words = [word.lower() for word in words if len(word) >= 3]

	return words

def loadEmailData():
	emailData = [] ; emailClasses = [] ; fullText = []
	for i in range(1, 26):
		content = open('naiveBayes/email/ham/' + str(i) + '.txt').read()
		words = parseTxt(content)
		emailData.append(words)
		emailClasses.append(0)
		fullText.extend(words)

		content = open('naiveBayes/email/spam/' + str(i) + '.txt').read()
		words = parseTxt(content)
		emailData.append(words)
		fullText.extend(words)
		emailClasses.append(1)
	return emailData, emailClasses, fullText

# return the trainSet and testSet's index stochastically
# 10 testing data(doc)
def createTrainAndTestSet(dataSet):
	trainSet = list(range(50)) ; testSet = []
	# print(trainSet)
	for i in range(10):
		randIndex = np.random.randint(0, len(trainSet))
		# print(randIndex)
		testSet.append(trainSet[randIndex])
		del(trainSet[randIndex])
	return trainSet, testSet

def spamTest():
	docList, classList, fullText = loadEmailData()
	vocabList = createVacabList(docList)
	error = 0 ; count = 0

	# get the index of trainSet and testSet
	trainSet, testSet = createTrainAndTestSet(docList)
	# print(trainSet, testSet)

	trainMatrix = [] ; trainClasses = []
	for docIndex in trainSet:
		trainMatrix.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	print(trainSet, trainClasses)

	p0V, p1V, pSp = trainNB0(np.array(trainMatrix), np.array(trainClasses))	

	for docIndex in testSet:
		testMatrix = bagOfWords2VecMN(vocabList, docList[docIndex])

		preRes = classifyNB(testMatrix, p0V, p1V, pSp)
		if classList[docIndex] != preRes:
			error += 1
		
		count += 1

		print('the %dth doc classified as: %d, and its real class is: %d' % (docIndex, preRes, classList[docIndex]))	
	print('the error rate is: ', error/float(count))

# return the top 30 words back because these have much big part for the whole TEXT!
def	clacMostFreq(vocabList, fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	
	sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
	
	return sortedFreq[:30]

# gee the wordList of feed1/0's 'summary' of entries'
# and added it into docList returned
def localWords(feed1, feed0):
	docList = [] ;  classList = [] ; fullText = []

	minLen = min(len(feed1['entries']), len(feed0['entries']))

	for i in range(minLen):
		wordList = parseTxt(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		
		wordList = parseTxt(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList = createVacabList(docList)
	top30Words = clacMostFreq(vocabList, fullText)
	# print(top30Words)

	for pairW in top30Words:
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])

	trainingSet = list(range(2*minLen)) ; testSet = []

	for i in range(20):
		randIndex = np.random.randint(0, len(trainingSet))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMatrix = [] ; trainClasses = []

	for docIndex in trainingSet:
		trainMatrix.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[randIndex])
	
	p0V, p1V, pSpam = trainNB0(trainMatrix, trainClasses)
	error = 0 ; count = 0

	for docIndex in testSet:
		testData = setOfWords2Vec(vocabList, docList[docIndex])
		preRes = classifyNB(testData, p0V, p1V, pSpam)

		if classList[docIndex] != preRes: error += 1
		count += 1
		
		print('the %dth doc classified as: %d, and its real class is: %d' % (docIndex, preRes, classList[docIndex]))	
	print('the error rate is: ', error/float(count))
	return vocabList, p0V, p1V

def getTopWords(ny, sf):
	import operator
	vocabList, p0V, p1V = localWords(ny, sf)
	topNY = [] ; topSF = []

	for i in range(len(p0V)):
		if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
		if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
	sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
	print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF')
	for item in sortedSF:
		print(item[0])

	sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
	print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY')
	for item in sortedNY:
		print(item[0])

def main():
	# testingNB()
	# spamTest()
	ny = feedparser.parse('https://newyork.craigslist.org/search/stp?format=rss')
	sf = feedparser.parse('https://sfbay.craigslist.org/search/stp?format=rss')

	# localWords(ny, sf)
	getTopWords(ny, sf)

if __name__ == '__main__':
	main()
