from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['a','a','b','b']
	return group, labels

def classify(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0] # shqpe 返回矩阵的（4，2） dataSetSize = 4

	# distance
	print(tile(inX,(dataSetSize,1)))
	diffMat = tile(inX,(dataSetSize,1)) -dataSet # plat inX
	print("diffMat",diffMat)

	sqDiffMat = diffMat**2 # 2次幂
	print(sqDiffMat)
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	print("distance",distances)
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		print(voteIlabel)
		classCount[voteIlabel] = classCount.get(voteIlabel ,0)+1
		sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),
				reverse=True)
	return sortedClassCount[0][0]

def clfy(point):
	group,labels = createDataSet()
	label = classify(point,group,labels,3)
	return point,label
clfy([0,0])
