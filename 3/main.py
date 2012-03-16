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
		rez[cn] = [cl(j, Orange.classification.Classifier.GetProbabilities) for j in testData]

	cPickle.dump(rez, file("rezPickled/RF200.pickled", "w"), -1)
	
def buildKNN(k=0):
	knn = Orange.classification.knn.kNNLearner()
	knn.k = k
	knn.distance_constructor = Orange.distance.Manhattan()
	rez = {}
	for i, cn in enumerate(trainClasses):
	#cn = "c40"
		sys.stdout.flush()
		sys.stdout.write("\r%3d%% done, current class: %s" % (100.0*i/82, cn))
		cl = knn(trainSingleClass(cn))
		rez[cn] = [cl(j, Orange.classification.Classifier.GetProbabilities) for j in testData]

	cPickle.dump(rez, file("rezPickled/KNN%dM.pickled" % k, "w"), -1)

def wanabeknn(k=15):
	from collections import Counter
	ftrd = open("minidata/trainingData.csv")
	fted = open("minidata/testData.csv")
	flab = open("minidata/trainingLabels.csv")

	lab = [[int(j) for j in i.strip().split(",")]  for i in flab.readlines()]
	trd = [[int(j) for j in i.strip().split("\t")] for i in ftrd.readlines()]
	ted = [[int(j) for j in i.strip().split("\t")] for i in fted.readlines()]

	def dist(a,b): return sum([min(a[i], b[i]) for i in xrange(len(a))])

	rez = []
	for v in ted:
		print "hurej  %4d   %3d" % ( len(rez),len(rez[-1:]))
		t = []
		for trindex, train in enumerate(trd):
			t.append((dist(train, v), trindex))
		tt = sorted(t, reverse=True)
		ll = []
		for i in range(k): ll += lab[tt[i][1]]
		n = len(ll)
		for i in range(k/3): ll += lab[tt[i][1]]
		rez.append([x[0] for x in Counter.most_common(Counter(ll),n/k)])
		print rez
	cPickle.dump(rez, file("rezPickled/wnbknn%d.pickled" % k, "w"), -1)

def buildBayes(l=200):
	s = Orange.preprocess.Preprocessor_featureSelection(limit=l, measure=Orange.feature.scoring.InfoGain())
	bayes = Orange.classification.bayes.NaiveLearner()
	
	rez = {}
	for cn in trainClasses:
		cl = bayes(s(trainSingleClass(cn)))
		rez[cn] = [cl(j) for j in testData]
	
	cPickle.dump(rez, file("rezPickled/bayesl%d.pickled" % l, "w"), -1)

def convertToCsv1():
	rez = cPickle.load(file("rezPickled/RF200.pickled"))
	t = [[ii for ii,dd in rez.items() if dd[i]['T'] > .3] for i in range(2000)]
	print sorted([int(t[ii][1][1:]) for ii in range(len(t))])

def convertToCsv2():
	rez = cPickle.load(file("rezPickled/RF200.pickled"))
	for i in xrange(len(testData)):
		t = sorted([(dd[i]['T'], ii) for ii, dd in rez.items() if dd[i]['T']>0.23679999999999999999999999999999999999999], reverse=True)
		if t == []:
			t = [(0,"c40")]
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


def convertToCsv5():
	rez = cPickle.load(file("rezPickled/KNN0.pickled"))
	for i in xrange(len(testData)):
		t = sorted([(dd[i]['T'], ii) for ii, dd in rez.items() if dd[i]['T']>0.2244], reverse=True)
		if t == []:
			t = [(0,"c40")]
		print sorted([int(t[ii][1][1:]) for ii in range(len(t))])

def convertToCsv6():
	rez = cPickle.load(file("rezPickled/KNN0M.pickled"))
	for i in xrange(len(testData)):
		t = sorted([(dd[i]['T'], ii) for ii, dd in rez.items() if dd[i]['T']>0.2258], reverse=True)
		if t == []:
			t = [(0,"c40")]
		print sorted([int(t[ii][1][1:]) for ii in range(len(t))])

def ctc():
	rez = cPickle.load(file("rezPickled/RF200.pickled"))

	from collections import Counter
	fl = open("minidata/trainingLabels.csv")
	t = [i.strip().split(',') for i in fl]
	c = Counter()
	for i in t:
		c.update(i)
	fl.close()	
	
	csv = [[] for i in range(2000)]
	for i in trainClasses:
		for j in sorted([(xx['T'], ii) for ii,xx in enumerate(rez[i])], reverse=True)[:c[i[1:]]]:
			csv[j[1]].append(int(i[1:]))

	for i in csv:
		print i
