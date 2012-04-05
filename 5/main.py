import numpy as np
from copy import deepcopy

import Orange
import numpy as np
data = Orange.data.Table("housing")
x, y, _ = data.to_numpy()
x = x[:,3]

#x = np.array([1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83])
#y = np.array([52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46])

X = np.vstack([np.ones(x.shape[0]), x]).T
Y = y

def iterLoop(end=-1):
	i = 0
	while i != end:
		i+=1
		yield i

def gradient(X,Y,alfa=0.001, maxIt=-1, eps=1e-5):
	tlx = []
	tly = []
	m, n = X.shape
	theta = np.array([-100. for i in range(n)])
	otheta = deepcopy(theta)
	for itc in iterLoop(maxIt):
		tlx.append(theta[0])
		tly.append(theta[1])
		for j in range(n):
			theta[j] = theta[j] + alfa * sum([(Y[i] - theta.dot(X[i,:]))*X[i,j] for i in range(m)])
		#print abs(sum(theta-otheta))
		if(abs(sum(theta-otheta)) < eps):
			return theta, itc, tlx, tly
		else:
			otheta = deepcopy(theta)
	return theta, itc, tlx, tly

a,i,tlx,tly = gradient(X,Y)
print a, i
import pylab
#aa = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
#pylab.plot(x,y,'.')
#pylab.plot(x, a.dot(X.T))
#pylab.show()

from pylab import *
xlist = linspace(-100, 100, 1000)
ylist = linspace(-100, 100, 1000)
X, Y = meshgrid(xlist, ylist)
Z = zeros(X.shape)
for i in range(x.size):
	Z += (X + Y*x[i] - y[i])**2
Z /= 2
contour(X, Y, log(Z))
plot(tlx, tly, 'bx-')
plot(tlx[-1], tly[-1], 'ro')
show()

#import Orange
#import numpy as np
#from copy import deepcopy

#data = Orange.data.Table("housing")
#x, y, _ = data.to_numpy()

##samo za primer
#x = x[:,5]
#X = np.vstack([np.ones(x.shape[0]), x]).T
#Y = y

#a,i = gradient(X,Y)
#print i
#import pylab
##aa = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
#pylab.plot(x,y,'.')
#pylab.plot(x, a.dot(X.T))
#pylab.show()

#def analiticna(X,Y):
	#X = np.matrix(X)
	#Y = np.matrix(Y)
	#return (X.T * X).I * X.T * Y

#def hi(X, i, theta):
	#return theta * X[i,:].T
	
#def gradient(X,Y,alfa=0.0001, eps=0.00000001):
	#m, n = X.shape
	#X = np.hstack((np.matrix(np.ones(m)).T, X)) #dodamo enko za x0
	#n += 1 #ker smo dodali stolpec
	#theta = np.matrix([1 for i in range(n)], dtype=float) #zacetna theta je 1
	#otheta = np.matrix([1 for i in range(n)], dtype=float) #zacetna theta je 1
	#for x in range(100):
		#for j in range(n):
			#theta[0,j] = theta[0,j] + alfa * sum([(Y[i,0] - hi(X, i, theta))*X[i,j] for i in range(m)])
		#if(abs(float(sum(theta.T-otheta.T))) < eps):
			#return theta
		#else:
			##print abs(float(sum(theta.T-otheta.T)))
			#otheta = deepcopy(theta)
	#return theta
			
#import pylab
#a = analiticna(X,Y)
#print a
##a = gradient(X,Y)
##print a
##aa = [a[0,0] + i*a[0,1] for i in range(100)]
#pylab.plot(X,Y,'*')
#pylab.plot(X, X*a[1,0] + a[0,0])
#pylab.show()