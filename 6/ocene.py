import Orange
import numpy as np

def cvkf(learner, data, k):
	n = len(data)
	selection = [int(i/(float(n)/k)) for i in xrange(n)]
	res = []
	for i in xrange(k):
		cl = learner(data.select(selection, i, negate=1))
		res += [cl(t, Orange.classification.Classifier.GetProbabilities)[True] for t in data.select(selection, i)]
	return res

def logLoss(T,P):
	return -1./T.size*sum(T*np.log(P)+(1-T)*np.log(1-P))

def crossValidation(learners, data, k=5):
	_, t, _ = data.to_numpy()
	for i in learners:
		p = cvkf(i, data, k)
		print "%10s: %5.3f" % (i.name, logLoss(t,p))

data = Orange.data.Table("data/train.tab")
lr = Orange.classification.logreg.LogRegLearner(remove_singular=1)
crossValidation([lr], data)