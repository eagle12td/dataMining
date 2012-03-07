import sys
import Orange
import cPickle

trainData = cPickle.load(file("minidata/trainingDataO.pickled"))
testData = cPickle.load(file("minidata/testDataO.pickled"))

trainClasses = {v.name:v for v in trainData.domain.class_vars}
trainSingleClass = lambda x: Orange.data.Table(Orange.data.Domain(trainData.domain.features + [trainClasses[x]]), trainData)
testData = Orange.data.Table(Orange.data.Domain(trainData.domain.features + [Orange.feature.Discrete("class", values=["F","T"])]), testData)

forest = Orange.ensemble.forest.RandomForestLearner(trees=200, name="forest")
rez = {}
for i, cn in enumerate(trainClasses):
#cn = "c40"
	sys.stdout.write("%3d%% done, current class: %s" % (100.0*i/82, cn))
	cl = forest(trainSingleClass(cn))
	rez[cn] = [cl(i, Orange.classification.Classifier.GetProbabilities) for i in testData]

cPickle.dump(rez, file("vseAliNic", "w"), -1)
