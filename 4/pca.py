import Orange
import cPickle
from time import time

#bolje bi bilo (po mojem) ce bi to nardil kar cez celo tabelo (10k A)
trainData = cPickle.load(file("minidata/trainingDataO.pickled"))
testData = cPickle.load(file("minidata/testDataO.pickled"))

pcaTable = Orange.data.Table(trainData)
[pcaTable.append(i) for i in testData]
pca = Orange.projection.linear.Pca(Orange.data.Table(pcaTable))
pcaTrainData = pca(trainData)
pcaTestData = pca(testData)

trainClasses = {v.name:v for v in trainData.domain.class_vars}
def pcaTrainSingleClass(c):
	domain = Orange.data.Domain(pcaTrainData.domain.features + [trainClasses[c]])
	table = Orange.data.Table(domain, pcaTrainData)
	for i in xrange(len(pcaTrainData)):
		table[i][-1] = trainData[i][c]
	return table

#learner = Orange.ensemble.forest.RandomForestLearner(trees=200, name="forest")

#res = [{} for i in xrange(len(pcaTestData))]
#for cn in trainClasses:
#	model = learner(pcaTrainSingleClass(cn))
#	for index, row in enumerate(pcaTestData):
#		res[index][cn] = model(row, Orange.classification.Classifier.GetProbabilities)["T"]

#cPickle.dump(res, file("rezPickled/%s_%d.pickled" % (learner.name, time()), "w"), -1)
