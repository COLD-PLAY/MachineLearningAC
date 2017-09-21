import math
import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(filename):
    dataSet = []

    fr = open(filename, 'r')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # fltLine = map(float, curLine)
        fltLine = []
        for i in range(len(curLine)):
            fltLine.append(float(curLine[i]))

        # print(fltLine)
        dataSet.append(fltLine)
    
    return np.mat(dataSet)

# algorithm kernel
def kMeans(dataSet, k):
    m, n = np.shape(dataSet)
    Mu = initMu(dataSet, k)

    # print(Mu)

    C = minDis(dataSet, Mu)
    C_ = np.zeros((m, 1))
    
    # for i in range(1000):
    #     Mu = descent(dataSet, Mu, C)
    #     C = minDis(dataSet, Mu)    

    count = 0
    while (C_ != C).any():
        count += 1
        C_ = C
        Mu = descent(dataSet, Mu, C)
        C = minDis(dataSet, Mu)    
        
    plotData(dataSet, C, Mu, count)

# find the optimization of Mu to make the total distance minimization
def descent(dataSet, Mu, C):
    k, n = Mu.shape
    m = C.shape[0]

    Mu_n = np.mat(np.zeros((k, n)))
    for j in range(k):
        MuJ = np.mat(np.zeros((1, n)))
        count = 0
        for i in range(m):
            if C[i, 0] == j:
                count += 1
                MuJ += dataSet[i, :]
        Mu_n[j, :] = MuJ/count
    return Mu_n
            
# calcute distance of vecA and vecB
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

# init the Mu randomly between the minX and maxX
def initMu(dataSet, k):
    m, n = dataSet.shape

    Mu = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)

        Mu[:, j] = minJ + rangeJ*np.random.rand(k, 1)
    return Mu

# calculate the most close Mu[i] for each X[i]
def minDis(X, Mu):
    m = np.shape(X)[0]
    k = np.shape(Mu)[0]

    C = np.mat(np.zeros((m, 1)))

    for i in range(0, m):
        min_d = np.inf
        for j in range(k):
            # dis = (X[i, :] - Mu[j, :]) * (X[i, :] - Mu[j, :]).T
            dis = distEclud(Mu[j, :], X[i, :])
            if min_d > dis:
                min_d = dis
                C[i, 0] = j
    return C

# plot the data which had been classified to k's kinds
def plotData(dataSet, C, Mu, i=100):
    k = Mu.shape[0]
    m, n = dataSet.shape

    plt.title('i: %d' % i)
    for i in range(m):
        plt.plot(dataSet[i, 0], dataSet[i, 1], 'C%do' % C[i])
        
    for i in range(k):
        plt.plot(Mu[i, 0], Mu[i, 1], 'C%d+' % i)
    plt.show()

if __name__ == '__main__':
    dataSet = loadDataSet('kMeans/testSet.txt')
    kMeans(dataSet, 9)
    # plotData(dataSet)

    # print(dataSet)

    # a = np.mat([[1, 2, 3], [4, 5, 6]])
    # b = np.mat([[1, 2, 3], [4, 5, 7]])
    # for i in range(0, 2):
    # 	c = a[i, :] - b[i, :]
    # 	print(c)
    
    # bino = np.random.binomial(40, 0.5, size=100)
    # # bino = sorted(bino)
    # # print(bino)

    # def count(num, n):
    #     c = 0
    #     for i in bino:
    #         if num == i: c += 1
        
    #     return c / float(n)

    # distribution = [count(i, 40) for i in range(41)]
    # print(distribution)
    # plt.plot(distribution, 'ro')
    # plt.show()