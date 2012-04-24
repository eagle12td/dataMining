import Orange
from collections import Counter
import numpy as np

#np.set_printoptions(threshold='nan')


data = Orange.data.Table("data/train.tab")
X, Y, _ = data.to_numpy()
n, m = X.shape
print "St. atributov: ", m
print "St. primerov: ", n

print "Gostota matrike (nenicleni elementi): %.2f%%" % (1.*sum(sum(X!=0))/X.size)

print "nicle po atributih", sum(X==0)

nrVal = np.asarray([np.unique(X[:,i]).size for i in range(X.shape[1])])

print "st. razlicnih vrednosti po atributih", nrVal

print "polni atributi: ", np.nonzero(sum(X!=0)==n)[0]
for i in range(200):
	print "prazni atributi (%d nenicleni element): %d" % (i, np.nonzero(sum(X==0)==n-i)[0].size)

a = sum(X!=0)
print a.min(), a.argmin()
print a.max(), a.argmax()




print np.nonzero(nrVal<10)[0]

#print [rf(attr, data) for attr in data.domain.attributes]


#import random
#alfa = 0.05
#shuffleCount = 500
#gain = Orange.feature.scoring.Relief()
#ig = np.array([gain(feature, data) for feature in data.domain.features])
#suma = np.zeros(len(data.domain.features))
#for s in xrange(shuffleCount):
	#c = [d.get_class() for d in data]
	#random.shuffle(c)
	#[d.set_class(c[i]) for i,d in enumerate(data)]

	#igp = np.array([gain(feature, data) for feature in data.domain.features])
	#suma += np.greater(igp, ig) 

#suma /= shuffleCount
#np.save("suma", suma)
#t = np.nonzero(suma<alfa)
#np.save("dobri", t)

def logLoss(T,P):
	return -1./T.size*sum(T*np.log(P)+(1-T)*np.log(1-P))
