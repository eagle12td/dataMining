import Orange
from collections import Counter
import numpy as np

#np.set_printoptions(threshold='nan')

data = Orange.data.Table("data/train.tab")
X, Y, _ = data.to_numpy()
n, m = X.shape

print "St. atributov: ", m
print "St. primerov: ", n

print "min element: ", X.min()
print "max element: ", X.max()

print "Gostota matrike (nenicleni elementi): %.2f%%" % (100.*sum(sum(X!=0))/X.size)

print "nicle po atributih", sum(X==0)

nrVal = np.asarray([np.unique(X[:,i]).size for i in range(X.shape[1])])
print "st. razlicnih vrednosti po atributih", nrVal
#print "polni atributi: ", np.nonzero(sum(X!=0)==n)[0]
print "st. polnih atributov: ", len(np.nonzero(sum(X!=0)==n)[0])
for i in range(10):
	print "prazni atributi (%d neniclenih elementov): %d" % (i, np.nonzero(sum(X==0)==n-i)[0].size)

a = sum(X!=0)
print "minimalna populacija izmer vseh atributov: ", a.min()#, a.argmin()
print "maximalna populacija izmed vseh atributov: ", a.max() #, a.argmax()

print "Deskritiziramo lahko (za max %d elementov) %d atributov" % (10, len(np.nonzero(nrVal<10)[0]))