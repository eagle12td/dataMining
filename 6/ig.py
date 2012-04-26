import Orange
import numpy as np

def mpi_entropy(y, base=2):
	"""Calculate entropy of a discrete distribution/histogram
	   (C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>
	"""
	invtotal = 1./float(len(y))
	p = np.array([invtotal*sum(y==l) for l in np.unique(y)])
	S = -1.0*sum(p*np.log(p))/np.log(base)
	return S

def condentropy(truelabels, labels):
	"""Calculate conditional entropy of one label distribution given another label distribution
	   (C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>
	"""
	labels=np.array(labels)
	truelabels=np.array(truelabels)
	condent=0.
	for l in xrange(min(labels),max(labels)+1):
		sublabels = truelabels[ labels==l ]
		condent += len(sublabels)*mpi_entropy( sublabels )
	return condent/float(len(labels))

def IG(x,y):
	return mpi_entropy(x) - condentropy(x,y)

if __name__ == "__main__":
	data = Orange.data.Table("data/train.tab")
	X, Y, _ = data.to_numpy()
	n, m = X.shape #n = st. primerov; m = st. atributov

	X *= 10

	X = np.array(X, dtype=int)
	Y = np.array(Y, dtype=int)
	
	ig = np.array([IG(X[:,i], Y) for i in xrange(m)])

	nrPerm = 100
	alfa = 0.05
	
	suma = np.zeros(m)
	for p in xrange(nrPerm):
		np.random.shuffle(Y)
		igp = np.array([IG(X[:,i], Y) for i in xrange(m)])
		suma += np.greater(igp, ig)
	suma /= nrPerm
	np.save("ig%d" % (nrPerm), np.nonzero(suma<alfa)[0])
