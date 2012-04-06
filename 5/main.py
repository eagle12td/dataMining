import numpy as np
import pylab as pl
import copy
import Orange

zt = -50.

data = Orange.data.Table("housing")
X, Y, _ = data.to_numpy()

def iterLoop(end=-1):
	i = 0
	while i != end:
		i+=1
		yield i

def analiticna(X,Y):
	return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def normalPlot(X, Y, index=0, alfa=0.001, eps=1e-5, maxIt=-1):
	pl.close()
	X = np.column_stack([np.ones(X.shape[0]), X[:,index]])
	xlist = np.linspace(-100, 100, 100)
	ylist = np.linspace(-100, 100, 100)
	XX, YY = np.meshgrid(xlist, ylist)
	ZZ = np.zeros(XX.shape)
	for i in range(X.shape[0]):
		ZZ += (XX + YY*X[i,1] - Y[i])**2
	ZZ /= 2
	pl.contour(XX, YY, ZZ, np.linspace(ZZ.min(), ZZ.min()*1000, 100))

	m, n = X.shape
	theta = np.array([zt for i in range(n)])
	otheta = copy.deepcopy(theta)
	for itc in iterLoop(maxIt):
		for i in range(m):
			for j in range(n):
				theta[j] = theta[j] + alfa * (Y[i] - theta.dot(X[i,:]))*X[i,j]
		if(abs(sum(theta-otheta)) < eps):
			break
		pl.plot([otheta[0],theta[0]] ,[otheta[1], theta[1]], 'kx-')
		otheta = copy.deepcopy(theta)
	r = analiticna(X,Y)
	pl.plot(theta[0], theta[1], 'ro')
	pl.plot([-100,100],[r[1]]*2,'k-')
	pl.plot([r[0]]*2,[-100,100],'k-')
	pl.show()
	pl.close()
	pl.plot(X[:,1], Y, '*')
	pl.plot(X[:,1], X.dot(theta))
	pl.plot(X[:,1], X.dot(r))
	pl.show()
	
def batchPlot(X, Y, index=0, alfa=0.001, eps=1e-5, maxIt=-1):
	pl.close()
	X = np.column_stack([np.ones(X.shape[0]), X[:,index]])
	xlist = np.linspace(-100, 100, 100)
	ylist = np.linspace(-100, 100, 100)
	XX, YY = np.meshgrid(xlist, ylist)
	ZZ = np.zeros(XX.shape)
	for i in range(X.shape[0]):
		ZZ += (XX + YY*X[i,1] - Y[i])**2
	ZZ /= 2
	pl.contour(XX, YY, ZZ, np.linspace(ZZ.min(), ZZ.min()*1000, 100))

	m, n = X.shape
	theta = np.array([zt for i in range(n)])
	otheta = copy.deepcopy(theta)
	for itc in iterLoop(maxIt):
		for j in range(n):
			theta[j] = theta[j] + alfa * sum([(Y[i] - theta.dot(X[i,:]))*X[i,j] for i in range(m)])
		if(abs(sum(theta-otheta)) < eps):
			break
		pl.plot([otheta[0],theta[0]] ,[otheta[1], theta[1]], 'kx-')
		otheta = copy.deepcopy(theta)
	r = analiticna(X,Y)
	pl.plot(theta[0], theta[1], 'ro')
	pl.plot([-100,100],[r[1]]*2,'k-')
	pl.plot([r[0]]*2,[-100,100],'k-')
	pl.show()
	pl.close()
	pl.plot(X[:,1], Y, '*')
	pl.plot(X[:,1], X.dot(theta))
	pl.plot(X[:,1], X.dot(r))
	pl.show()

batchPlot(X, Y, 7, 0.0001, 1e-5)
#normalPlot(X, Y, 7, alfa=0.001, eps=1e-5)
#normalPlot(X, Y, 7, alfa=0.0001, eps=1e-5)
#normalPlot(X, Y, 7, alfa=0.00001, eps=1e-5)