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
	P = P*0.9998+0.0001
	return -1./T.size*sum(T*np.log(P)+(1-T)*np.log(1-P))

def crossValidation(learners, data, k=10):
	_, t, _ = data.to_numpy()
	for i in learners:
		p = cvkf(i, data, k)
		print "%10s: %5.3f" % (i.name, logLoss(t,p))

def findBestConst(data):
	_,yTrue,_ = data.to_numpy()
	m = yTrue.size
	yPred = np.array([0.1]*m)
	ll = logLoss(yTrue, yPred)
	prev = ll
	best = 0
	for i in range(2000,7000):
		a = i/10000.0
		yPred = np.array([a]*m)
		ll = logLoss(yTrue, yPred)
		#print "%.6f   %.10f      %d" % (a,ll, int(prev<ll))
		if (prev<ll):
			best = a
			break;
		prev = ll
	print "%10s: %5.3f" % ("best const", logLoss(yTrue,yPred))

if __name__ == "__main__":
	data = Orange.data.Table("data/train.tab")
	lr = Orange.classification.logreg.LogRegLearner(remove_singular=1, name="logreg")
	rf = Orange.ensemble.forest.RandomForestLearner(trees=50, name="forest50")
	nb = Orange.classification.bayes.NaiveLearner(name='bayes')
	knn = Orange.classification.knn.kNNLearner(name="knn")
	lsvm = Orange.classification.svm.LinearSVMLearner(name="linearSVM")
	crossValidation([lr, rf, nb, knn, lsvm], data)

	findBestConst(data) #const: rez = 0.542400