Problem 1:

1.KNN:

My code is as following:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import heapq
from collections import Counter

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
mnist["data"], mnist["target"]

def knn(test,k=5,ord=1):
    train = mnist["data"][0:1000]
    target = mnist["target"][0:1000]
    distances=[]
    for item in train:
        distance=np.linalg.norm(test-item,ord=ord)
        distances.append(distance)
    min_num_index_list = map(distances.index, heapq.nsmallest(k, distances))
    targets = []
    for i in min_num_index_list:
        targets.append(target[i])
    return Counter(targets).most_common()[0][0]


```

KNN function will return the most possible number.

And then I use the following code to test it:

```python
for ord in range(1,4):
    counter = 0
    for item in range(1001,2000):
        if knn(mnist["data"][item],k=5,ord=ord) == mnist["target"][item]:
            counter+=1
    print ("when ord = ",ord,"correct rate =",counter/1000)
```

I use 1000 items to train my code and then use 1000 to test it. The result depend on the ord:

```python
when ord =  1 correct rate = 0.853
when ord =  2 correct rate = 0.865
when ord =  3 correct rate = 0.871
```

We can find that when ord = 3 is the most accurate one but these 3 not differ a lot.

And among 1000-2000, there are 100 9s and the our function give 128 9s. Among 128 9s, the correct rate is 76%.



2.Decision Tree

```python
from numpy import *
import operator
 
def calcShannonEntropy(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featureVec in dataSet:
		currentLabel = featureVec[0]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 1
		else:
			labelCounts[currentLabel] += 1
	shannonEntropy = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEntropy -= prob  * log2(prob)
	return shannonEntropy
 
#get all rows whose axis item equals value.
def splitDataSet(dataSet, axis, value):
	subDataSet = []
	for featureVec in dataSet:
		if featureVec[axis] == value:
			reducedFeatureVec = featureVec[:axis]
			reducedFeatureVec.extend(featureVec[axis+1:])	#if axis == -1, this will cause error!
			subDataSet.append(reducedFeatureVec)
	return subDataSet
 
def chooseBestFeatureToSplit(dataSet):
	#Notice: Actucally, index 0 of numFeatures is not feature(it is class label).
	numFeatures = len(dataSet[0])	
	baseEntropy = calcShannonEntropy(dataSet)
	bestInfoGain = 0.0
	bestFeature = numFeatures - 1 	#DO NOT use -1! or splitDataSet(dataSet, -1, value) will cause error!
	#feature index start with 1(not 0)!
	for i in range(numFeatures)[1:]:
		featureList = [example[i] for example in dataSet]
		featureSet = set(featureList)
		newEntropy = 0.0
		for value in featureSet:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEntropy(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
 
#classify on leaf of decision tree.
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]
 
#Create Decision Tree.
def createDecisionTree(dataSet, features):
#	print ((create decision tree... length of features is:)+str(len(features))
	classList = [example[0] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeatureIndex = chooseBestFeatureToSplit(dataSet) 
	bestFeatureLabel = features[bestFeatureIndex]
	myTree = {bestFeatureLabel:{}}
	del(features[bestFeatureIndex])
	featureValues = [example[bestFeatureIndex] for example in dataSet]
	featureSet = set(featureValues)
	for value in featureSet:
		subFeatures = features[:]	
		myTree[bestFeatureLabel][value] = createDecisionTree(splitDataSet(dataSet, bestFeatureIndex, value), subFeatures)
	return myTree
 
def line2Mat(line):
	mat = line.strip().split(' ')
	for i in range(len(mat)-1):	
		pixel = mat[i+1].split(':')[1]
		#change MNIST pixel data into 0/1 format.
		mat[i+1] = int(pixel)/128
	return mat
 
#return matrix as a list(instead of a matrix).
#features is the 28*28 pixels in MNIST dataset.
def file2Mat(fileName):
	f = open(fileName)
	lines = f.readlines()
	matrix = []
	for line in lines:
		mat = line2Mat(line)
		matrix.append(mat)
	f.close()
	return matrix
 
#Classify test file.
def classify(inputTree, featureLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featureIndex = featureLabels.index(firstStr)
	predictClass = '-1'
	for key in secondDict.keys():
		if testVec[featureIndex] == key:
			if type(secondDict[key]) == type({}):	
				predictClass = classify(secondDict[key], featureLabels, testVec)
			else:
				predictClass = secondDict[key]
	return predictClass
 
def classifyTestFile(inputTree, featureLabels, testDataSet):
	rightCnt = 0
	for i in range(len(testDataSet)):
		classLabel = testDataSet[i][0]
		predictClassLabel = classify(inputTree, featureLabels, testDataSet[i])
		if classLabel == predictClassLabel:
			rightCnt += 1 
	return float(rightCnt)/len(testDataSet)
 
def getFeatureLabels(length):
	strs = []
	for i in range(length):
		strs.append('#'+str(i))
	return strs
 
#Normal file
trainFile = 'train_1k.txt'	
testFile = 'test_1k.txt'
 
#train decision tree.
dataSet = file2Mat(trainFile)
#Actually, the 0 item is class, not feature labels.
featureLabels = getFeatureLabels(len(dataSet[0]))	
print("begin to create decision tree...")
myTree = createDecisionTree(dataSet, featureLabels)
print("create decision tree done.")
 
#predict with decision tree.	
testDataSet = file2Mat(testFile)
featureLabels = getFeatureLabels(len(testDataSet[0]))	
rightRatio = classifyTestFile(myTree, featureLabels, testDataSet)
print (rightRatio)
```



The result is correct rate is 75%. It performs worse than KNN algorithm.



3.Random Forests:

