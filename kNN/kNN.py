import numpy as np
import operator
import matplotlib.pyplot as plt

def createDataSet():
	group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	m, n = dataSet.shape
	diff = np.dot(np.ones((m, 1)), inX)/m - dataSet
	dis = (diff**2).sum(axis=1)**0.5

	sort_dis = dis.argsort()
	c = {}
	for i in range(k):
		v = labels[sort_dis[i]]
		c[v] = c.get(v, 0) + 1

	s = sorted(c.items(), key=operator.itemgetter(1), reverse=True)
	return s[0][0]

def file2matrix(filename):
	file = open(filename, 'r')

	dataSet = []
	labelSet = []

	for line in file.readlines():
		dataSet.append(list(map(lambda x: float(x), line.strip().split('\t')[:-1])))
		labelSet.append(int(line.strip().split('\t')[-1]))

	return np.array(dataSet), np.array(labelSet)

def plot(dataSet):
	figure = plt.figure()
	ax = figure.add_subplot(111)
	ax.scatter(dataSet[:, 1], dataSet[:, 2])
	
	plt.xlabel(u'玩视频游戏所耗时间百分比')
	plt.ylabel(u'每周所消费的冰淇淋公升数')

	plt.show()

if __name__ == '__main__':
	# group, labels = createDataSet()

	# label = classify0(np.array([[0.0, -0.2]]), group, labels, 2)
	# print(label)

	dataSet, labelSet = file2matrix('datingTestSet2.txt')

	# dataSet, labelSet = file2matrix('C:\\Users\\COLDPLAY\\Desktop\\GitHub\\MachineLearningAC\\LogisticRregression\\weatherTraining.txt')
	# plot(dataSet)

plt.plot((1, 2, 3), (4, 3, -1))
plt.xlabel(u'横坐标')
plt.ylabel(u'纵坐标')
plt.show()