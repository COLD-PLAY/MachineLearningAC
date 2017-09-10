import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(filename):
	dataSet = []

	fr = open(filename, 'r')
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataSet.append(fltLine)

	fr.close()
	# print(dataSet)
	return np.mat(dataSet)

def kMeans(dataSet, k):
	m, n = np.shape(dataSet)
	MU = np.mat(np.random.random((k, n)))

	print(MU)

	C = minDis(dataSet, MU)
	print(C)

def minDis(X, MU):
	m = np.shape(X)[0]
	k = np.shape(MU)[0]

	min_d = np.inf

	C = np.mat(np.zeros((m, 1)))

	for i in range(0, m):
		for j in range(0, k):
			dis = (X[i, :] - MU[j, :]) * (X[i, :] - MU[j, :]).T
			if min_d > dis:
				min_d = dis
				C[i, 0] = j
	return C

	
def plotData(dataSet):
	plt.plot(dataSet[:, 0], dataSet[:, 1], 'o')
	plt.show()

if __name__ == '__main__':
	dataSet = loadDataSet('testSet.txt')
	kMeans(dataSet, 4)
	# plotData(dataSet)

	# a = np.mat([[1, 2, 3], [4, 5, 6]])
	# b = np.mat([[1, 2, 3], [4, 5, 7]])
	# for i in range(0, 2):
	# 	c = a[i, :] - b[i, :]
	# 	print(c)