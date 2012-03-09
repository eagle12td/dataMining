import sys
import Orange
import cPickle

trainData = cPickle.load(file("minidata/trainingDataO.pickled"))
testData = cPickle.load(file("minidata/testDataO.pickled"))

trainClasses = {v.name:v for v in trainData.domain.class_vars}
trainSingleClass = lambda x: Orange.data.Table(Orange.data.Domain(trainData.domain.features + [trainClasses[x]]), trainData)
testData = Orange.data.Table(Orange.data.Domain(trainData.domain.features + [Orange.feature.Discrete("class", values=["F","T"])]), testData)

def buildRandomForest():
	forest = Orange.ensemble.forest.RandomForestLearner(trees=200, name="forest")
	rez = {}
	for i, cn in enumerate(trainClasses):
	#cn = "c40"
		sys.stdout.flush()
		sys.stdout.write("\r%3d%% done, current class: %s" % (100.0*i/82, cn))
		cl = forest(trainSingleClass(cn))
		rez[cn] = [cl(i, Orange.classification.Classifier.GetProbabilities) for i in testData]

	cPickle.dump(rez, file("rezPickled/RF200.pickled", "w"), -1)

def convertToCsv1():
	rez = cPickle.load(file("rezPickled/RF200.pickled"))
	t = [[ii for ii,dd in rez.items() if dd[i]['T'] > .3] for i in range(2000)]
	for i in t:
		print i

def convertToCsv2():
	rez = cPickle.load(file("rezPickled/RF200.pickled"))
	for i in xrange(len(testData)):
		t = sorted([(dd[i]['T'], ii) for ii, dd in rez.items()], reverse=True)[:4]
		print sorted([int(t[ii][1][1:]) for ii in range(len(t))])

def convertToCsv3():
	rez = cPickle.load(file("rezPickled/RF200.pickled"))

	from collections import Counter
	fl = open("minidata/trainingLabels.csv")
	t = [i.strip().split(',') for i in fl]
		c = Counter()
		for i in t:
				c.update(i)
	fl.close()

	for i,d in c.items():
		for j in rez["c"+str(i)]:
			j['T'] *= d
	for i in xrange(len(testData)):
		t = sorted([(dd[i]['T'], ii) for ii, dd in rez.items()], reverse=True)[:4]
		print sorted([int(t[ii][1][1:]) for ii in range(len(t))])


def convertToCsv4():
	MAX_CLASSES = 4
	MIN_CLASSES = 3
	rez = cPickle.load(file("rezPickled/RF200.pickled"))
	
	tt = []
	for i in xrange(len(testData)):
		t = sorted([(dd[i]['T'], ii) for ii, dd in rez.items()], reverse=True)[:MAX_CLASSES]
		tt.append(t[-1][0])
	import numpy as np
	th = np.median(tt)

	for i in xrange(len(testData)):
		tempMin = []
		tempMax = []
		for ii, dd in rez.items():
			if dd[i]['T'] > th:
				tempMax.append((dd[i]['T'], ii))
			else:
				tempMin.append((dd[i]['T'], ii))
		tMin = sorted(tempMin, reverse=True)[:MIN_CLASSES]
		tMax = sorted(tempMax, reverse=True)[:MAX_CLASSES]
		t = tMax
		for ii in range(MIN_CLASSES-len(tMax)):
			t.append(tMin[ii])
		print sorted([int(t[ii][1][1:]) for ii in range(len(t))])

convertToCsv4()
