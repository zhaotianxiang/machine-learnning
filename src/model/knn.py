import numpy as np
import matplotlib.pyplot as plt
import operator

def createDataSet():
	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['a','a','b','b']
	return group, labels

def file2matrix(filename):
	fr = open(filename)
	lines = fr.readlines()
	numberOfLines = len(lines)
	linesArray = lines[0].split(',')
	numberOfValue = len(linesArray)

	# init data set
	mat = np.zeros((numberOfLines-1,numberOfValue))
	index = 0
	for line in lines:
		line = line.strip()
		listFromLine = line.split(',')
		if(not listFromLine[0].replace(".",'').isdigit()):
			continue
		mat[index] = listFromLine
		index+=1
	print(mat)
	return mat

def normalize(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	diffDataSet = dataSet - np.tile(minVals,(m,1))
	normDataSet = diffDataSet/np.tile(ranges,(m,1))

	return normDataSet

def display(mat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(mat[:,1],mat[:2])
	plt.show()


def classify(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0] # shqpe 返回矩阵的（4，2） dataSetSize = 4
	diffMat = np.tile(inX,(dataSetSize,1)) -dataSet # plat inX
	sqDiffMat = diffMat**2 # 2次幂
	print(sqDiffMat)
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel ,0)+1
		sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),
				reverse=True)
	return sortedClassCount[0][0]

def clfy(point):
	group,labels = createDataSet()
	label = classify(point,group,labels,3)
	return point,label

data = file2matrix('../../data/knn/knn.txt')
print(normalize(data))

clfy([0,0])
