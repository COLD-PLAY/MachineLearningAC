# coding: utf-8

'''
Date: 2017-08-21 00:32
Author: Coldplay
logRegres.py
LOGISTIC REGRESSION ALGORITHM (DEEP LEARNING)
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import random

# 加载数据集
def loadDataSet():
	dataMat = []; labelMat = []

	fr = open('weatherTraining.txt')

	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))

	return np.mat(dataMat), np.mat(labelMat)
	# return dataMat, labelMat

# 逻辑函数
def sigmoid(inX):
	# print(inX)
	return 1.0/(1 + np.exp(-inX))

# 梯度上升算法
def gradAscent(dataMat, labelMat):
	labelMat = labelMat.transpose()
	m, n = np.shape(dataMat)

	alpha = 0.001
	maxCycles = 5000
	weights = np.ones((n, 1))

	for k in range(maxCycles):
		h = sigmoid(dataMat * weights)
		error = (labelMat - h)
		weights += alpha * dataMat.transpose() * error

	return weights

# 随机梯度上升算法
def stocGradAscent0(dataMat, labelMat):
    labelMat = labelMat.transpose()
    m, n = np.shape(dataMat)

    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))

    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights += alpha * dataMat.transpose()* error

    return weights

# 改进的随机梯度上升
def stocGradAscent1(dataMat, LabelMat, numIter=150):
	m, n = np.shape(dataMat)
	weights = np.ones((n, 1))

	for j in range(numIter):
		dataIndex = range(m)

		for i in range(m):
			alpha = 4/(1.0 + j + i) + 0.01		# alpha 每次回更新
			randIndex = int(random.uniform(0, len(dataIndex)))	# 随机选择更新

			h = sigmoid(sum(dataMat[randIndex] * weights))
			error = LabelMat[0, randIndex] - h

			weights += alpha * dataMat[randIndex].transpose() * error
			# dataMat = np.delete(dataMat, randIndex, 0)

	return weights

# 绘制最优直线
def plotBestFit(weights):
	dataMat, labelMat = loadDataSet()

	n = np.shape(dataMat)[0]

	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []

	for i in range(n):
		if int(labelMat[0, i]) == 1:
			xcord1.append(dataMat[i, 1]); ycord1.append(dataMat[i, 2])
		else:
			xcord2.append(dataMat[i, 1]); ycord2.append(dataMat[i, 2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	
	# x = np.arange(-3.0, 3.0, 0.1)
	# y = (-weights[0] - weights[1] * x)/weights[2]

	# ax.plot(x, y.transpose())
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5: return 1.0
	return 0.0

def loadDataSet_AC():
	ftTrain = open('horseColicTraining.txt')

	trainingSet = []; trainingLabels = []

	for line in ftTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []

		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))


	return np.mat(trainingSet), np.mat(trainingLabels)

def colicTest():
	trainingSet, trainingLabels = loadDataSet_AC()

	trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
	# print(trainWeights)			# 打印出训练结果的权重参数

	errorCount = 0; numTestVec = 0.0

	frTest = open('horseColicTest.txt')
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []

		for i in range(21):
			lineArr.append(float(currLine[i]))

		if int(classifyVector(np.mat(lineArr), np.mat(trainWeights))) != int(currLine[21]):
			errorCount += 1

	errorRate = float(errorCount)/numTestVec
	print('the error rate of this test is: %f' % errorRate)
	return errorRate

def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()

	print('after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests)))



if __name__ == '__main__':
	dataMat, labelMat = loadDataSet()
	# weights = gradAscent(dataMat, labelMat)
	# weights = stocGradAscent0(dataMat, labelMat)
	weights = stocGradAscent1(dataMat, labelMat, 150)

	print(weights)

	plotBestFit(weights)
	
	# colicTest()
	# multiTest()