import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def createDataSet():
	group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	m, n = dataSet.shape
	diff = np.tile(inX, (m, 1)) - dataSet
	dis = ((diff**2).sum(axis=1))**0.5

	sort_dis = dis.argsort()
	c = {}
	for i in range(k):
		v = labels[sort_dis[i]]
		c[v] = c.get(v, 0) + 1

	s = sorted(c.items(), key=operator.itemgetter(1), reverse=True)
	return s[0][0]

def file2matrix(filename):
	file = open(filename)

	dataSet = []
	labelSet = []

	for line in file.readlines():
		dataSet.append(list(map(lambda x: float(x), line.strip().split('\t')[:-1])))
		labelSet.append(int(line.strip().split('\t')[-1]))

	return np.array(dataSet), np.array(labelSet)

def image2vector(filename):
	returnVector = np.zeros((1, 1024))
	fr = open(filename)

	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVector[0, i*32 + j] = int(lineStr[j])
	return returnVector

def plot(dataSet):
	figure = plt.figure()
	ax = figure.add_subplot(111)
	ax.plot(dataSet[:, 1], dataSet[:, 2], style)

	# ax.axis([0.0, 25.0, 0.0, 2.0])
	
	plt.xlabel(u'玩视频游戏所耗时间百分比')
	plt.ylabel(u'每周所消费的冰淇淋公升数')

	plt.show()

def plot_classifier_by_12(dataSet, labelSet):
	dataSet_1 = []
	dataSet_2 = []
	dataSet_3 = []

	for i in range(np.shape(dataSet)[0]):
		if labelSet[i] == 1:
			dataSet_1.append(dataSet[i, :])
		elif labelSet[i] == 2:
			dataSet_2.append(dataSet[i, :])
		else:
			dataSet_3.append(dataSet[i, :])

	dataSet_1 = np.array(dataSet_1)
	dataSet_2 = np.array(dataSet_2)
	dataSet_3 = np.array(dataSet_3)

	figure = plt.figure()
	ax = figure.add_subplot(111)
	ax.plot(dataSet_1[:, 1], dataSet_1[:, 2], 'C1o', label='不喜欢')
	ax.plot(dataSet_2[:, 1], dataSet_2[:, 2], 'C2o', label='魅力一般')
	ax.plot(dataSet_3[:, 1], dataSet_3[:, 2], 'C3o', label='极具魅力')

	ax.legend()

	plt.xlabel(u'玩视频游戏所耗时间百分比')
	plt.ylabel(u'每周所消费的冰淇淋公升数')

	plt.show()

def plot_classifier_by_01(dataSet, labelSet):
	dataSet_1 = []
	dataSet_2 = []
	dataSet_3 = []

	for i in range(np.shape(dataSet)[0]):
		if labelSet[i] == 1:
			dataSet_1.append(dataSet[i, :])
		elif labelSet[i] == 2:
			dataSet_2.append(dataSet[i, :])
		else:
			dataSet_3.append(dataSet[i, :])

	dataSet_1 = np.array(dataSet_1)
	dataSet_2 = np.array(dataSet_2)
	dataSet_3 = np.array(dataSet_3)

	figure = plt.figure()
	ax = figure.add_subplot(111)
	ax.plot(dataSet_1[:, 0], dataSet_1[:, 1], 'C1o', label='不喜欢')
	ax.plot(dataSet_2[:, 0], dataSet_2[:, 1], 'C2o', label='魅力一般')
	ax.plot(dataSet_3[:, 0], dataSet_3[:, 1], 'C3o', label='极具魅力')

	ax.legend()

	plt.xlabel(u'每年获取的飞行常客里程数')
	plt.ylabel(u'玩视频游戏所耗时间百分比')

	plt.show()

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals

	normDataSet = np.zeros(np.shape(dataSet))

	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	normDataSet = normDataSet / np.tile(ranges, (m, 1))

	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)

	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0

	for i in range(numTestVecs):
		classifierResult = classify0(np.array([normMat[i, :]]), normMat[numTestVecs:m, :], \
			datingLabels[numTestVecs:m], 3)
		print('the classifier came back with: %d, the real answer is: %d'\
			% (classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]: errorCount += 1.0

	print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input(\
		'percentage of time spent playing video games?'))
	ffMiles = float(input(\
		'frequent flier miles earned per year?'))
	iceCream = float(input(\
		'liters of ice cream consumed per year?'))

	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)

	inX = np.array([[percentTats, ffMiles, iceCream]])
	classifierResult = classify0(inX, normMat, datingLabels, 3)

	print('You will probably like this person: ',\
		resultList[classifierResult - 1])

def handWritingClassTest():
	dataSet, labelSet = digits2matrix('trainingDigits')

	errorCount = 0

	print(dataSet.shape)

	for root, dirs, files in os.walk('trainingDigits'):
		m = len(files)
		for filename in files:
			ans = int(filename.split('_')[0])
			inX = image2vector(root + '/' + filename)

			classifierResult = classify0(inX, dataSet, labelSet, 7)
			print('the classifier came back with: %d, the real answer is: %d'\
				% (classifierResult, ans))

			if classifierResult != ans: errorCount += 1
	print('the total error rate is: %f' % (errorCount/float(m)))

def plot_digit(vector):
	figure = plt.figure()
	ax = figure.add_subplot(111)
	ax.axis([0.0, 32.0, 0.0, 32.0])

	m, n = vector.shape
	# print(m, n)
	for i in range(n):
		# ax.scatter(vector[0, i]//32, 32 - vector[0, i]%32, 'ro')
		if vector[0, i] == 1:
			ax.plot(i%32, 32 - i//32, 'ro')

	plt.show()

def digits2matrix(filename):
	for root, dirs, files in os.walk(filename):
		m = len(files)
		dataSet = np.zeros((m, 1024))
		labelSet = []

		for i in range(m):
			file = files[i]
			label = int(file.split('_')[0])
			labelSet.append(label)

			vector = image2vector(root + '/' + file)
			dataSet[i, :] = vector

	labelSet = np.array(labelSet)

	return dataSet, labelSet

def myHandWritingTest(jpgname):
	dataSet, labelSet = digits2matrix('trainingDigits')

	img = mpimg.imread(jpgname)

	vector = np.zeros((1, 1024))
	for i in range(32):
		for j in range(32):
			if sum(img[i, j, :]) < 700:
				vector[0, i*32 + j] = 1
			else:
				vector[0, i*32 + j] = 0
			print(int(vector[0, i*32 + j]), end='')
		print()

	# plot_digit(vector)
	ans = 2
	result = classify0(vector, dataSet, labelSet, 3)
	print('the classifier came back with: %d' % result)

if __name__ == '__main__':
	# group, labels = createDataSet()

	# label = classify0(np.array([[0.0, -0.2]]), group, labels, 2)
	# print(label)

	# dataSet, labelSet = file2matrix('datingTestSet.txt')

	# dataSet, labelSet = file2matrix('C:\\Users\\COLDPLAY\\Desktop\\GitHub\\MachineLearningAC\\LogisticRregression\\weatherTraining.txt')
	# plot(dataSet)
	# plot_classifier_by_12(dataSet, labelSet)
	# plot_classifier_by_01(dataSet, labelSet)

	# normDataSet = autoNorm(dataSet)[0]
	# plot_classifier_by_01(normDataSet, labelSet)
	# datingClassTest()
	# classifyPerson()
	handWritingClassTest()
	# for root, dirs, files in os.walk('trainingDigits'):
	# 	# for file in files:
	# 	for i in range(20):
	# 		vector = image2vector(root + '/' + files[i*100])

	# 		plot_digit(vector, i + 1)
	# myHandWritingTest('seven_32.jpg')