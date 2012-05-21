from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import Orange
import multiprocessing
import functools

MAXPROC = 8

def logLoss(T,P):
	T = np.array(T)
	P = np.array(P)
	return -1./T.size*sum(T*np.log(P)+(1-T)*np.log(1-P))

def cvNode(i, X, y, selection, lambda_):
	testX = X[selection==i]
	trainX = X[selection!=i]
	trainY = y[selection!=i]
	thetas = log_reg(trainX, trainY, lambda_)[0]
	return sigmoid(testX.dot(thetas))

def crossValidation(X, y, k=10, lambda_=0):
	n = len(y)
	selection = np.array([int(i/(float(n)/k)) for i in xrange(n)])
	pool = multiprocessing.Pool(MAXPROC)
	res = pool.map(functools.partial(cvNode, X=X, y=y, selection=selection, lambda_=lambda_), range(k))
	pool.close()
	pool.join()
	return np.concatenate(res)

def sigmoid(z): 
	return 1. / (1. + np.exp(-z))

def log_reg(X, y, lambda_):
	m = len(y)
	thetas0 = np.zeros(X.shape[1])
	def cost(thetas, X, y, lambda_):
		h = sigmoid(X.dot(thetas))
		y0 = np.log(1. - h[y==0])
		y1 = np.log(h[y==1])
		return -(sum(y0) + sum(y1))/m + lambda_ * sum(thetas**2)
	def grad(thetas, X, y, lambda_):
		h = sigmoid(X.dot(thetas))
		return (h - y).dot(X)/m + lambda_ * sum(2.*thetas)

	return fmin_l_bfgs_b(cost, thetas0, grad, args=(X, y, lambda_))

def feature_extend(X, n_features=1):
    return np.column_stack([X] + [X**(i+2) for i in range(n_features)])


if __name__ == "__main__":
	lambda_ = 0.01
	nrFolds = 5
	nrExtendFeatures = 1

	data = Orange.data.Table("data/train.tab")
	XX, YY, _ = data.to_numpy()
	XX = feature_extend(XX, nrExtendFeatures)

	napoved = crossValidation(XX, YY, nrFolds, lambda_)
	napoved = napoved*0.9998+0.0001
	print "lambda: %f\t logLoss: %f" % (lambda_, logLoss(YY, napoved))
