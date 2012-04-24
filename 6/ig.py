import Orange
import numpy as np

def mpi_entropy(y, base=2):
	"""Calculate entropy of a discrete distribution/histogram"""
	invtotal = 1./float(len(y))
	p = np.array([invtotal*sum(y==l) for l in np.unique(y)])
	S = -1.0*sum(p*np.log(p))/np.log(base)
	return S

def condentropy(truelabels, labels):
	"""Calculate conditional entropy of one label distribution given another label distribution"""
	labels=np.array(labels)
	truelabels=np.array(truelabels)
	condent=0.
	for l in xrange(min(labels),max(labels)+1):
		sublabels = truelabels[ labels==l ]
		condent += len(sublabels)*mpi_entropy( sublabels )
	return condent/float(len(labels))

def IG(x,y):
	return mpi_entropy(x) - condentropy(x,y)

data = Orange.data.Table("data/train.tab")
X, Y, _ = data.to_numpy()
n, m = X.shape #n = st. primerov; m = st. atributov

X.dtype = int
Y.dtype = int

ig = np.array([IG(X[:,i], Y) for i in xrange(m)])

nrPerm = 100
alfa = 0.05

suma = np.zeros(m)
for i in xrange(m):
	igp = np.array([IG(X[:,i], np.random.permutation(Y)) for p in xrange(nrPerm)])
	suma += np.greater(igp, ig)
suma /= nrPerm
np.save("ig%d" % (nrPerm), np.nonzero(suma<alfa)[0])