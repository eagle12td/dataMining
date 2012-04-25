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
	T = np.array(T)
	P = np.array(P)
	P = P*0.99999999998+0.00000000001
	return -1./T.size*sum(T*np.log(P)+(1-T)*np.log(1-P))

def crossValidation(learners, data, k=10):
	_, t, _ = data.to_numpy()
	for i in learners:
		p = cvkf(i, data, k)
		print "%10s: %5.3f" % (i.name, logLoss(t,p))

data = Orange.data.Table("data/train.tab")
lr = Orange.classification.logreg.LogRegLearner(remove_singular=1, name="logreg")
rf = Orange.ensemble.forest.RandomForestLearner(trees=50, name="forest")
knn = Orange.classification.knn.kNNLearner(name="knn")
lsvm = Orange.classification.svm.LinearSVMLearner(name="linearSVM")
crossValidation([lr, rf, knn], data)
