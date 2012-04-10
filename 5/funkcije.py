import numpy as np
import pylab as pl
import copy

plotsize = 20
zt = -20.
crange = np.exp(np.hstack([np.linspace(7.71, 9, 5),np.linspace(9, 13, 5)]))

def iterLoop(end=-1):
	i = 0
	while i != end:
		i+=1
		yield i

def plot(X, Y, theta, at, th):
	pl.close()
	[pl.plot(i[0], i[1], 'kx-') for i in th]
	
	pl.plot(theta[0], theta[1], 'ro')
	pl.plot([-plotsize,plotsize],[at[1]]*2,'k-')
	pl.plot([at[0]]*2,[-plotsize,plotsize],'k-')
	
	XX, YY = np.meshgrid(np.linspace(-plotsize, plotsize, 100), np.linspace(-plotsize, plotsize, 100))
	ZZ = np.zeros(XX.shape)
	for i in range(X.shape[0]):
		ZZ += (XX + YY*X[i,1] - Y[i])**2
	ZZ /= 2

	pl.clabel(pl.contour(XX, YY, ZZ, crange), inline=1, fontsize=8)
	pl.show()
	pl.close()
	pl.plot(X[:,1], Y, '*')
	pl.plot(X[:,1], X.dot(theta))
	pl.plot(X[:,1], X.dot(at))
	pl.show()

def normal(theta, alfa, X, Y):
	for i in range(X.shape[0]):
		theta = theta + alfa * (Y[i] - theta.dot(X[i]))*X[i]
	return theta

def batch(theta, alfa, X, Y):
	return theta + alfa * sum([(Y[i] - theta.dot(X[i]))*X[i] for i in range(X.shape[0])])

def getTheta(X, Y, index=-1, alfa=0.001, eps=1e-5, maxIt=-1, showplot=False, batchAlg=False):
	th = []
	if index == -1:
		if showplot:
			raise Exception("Plot is aviable only for one attribute")
		X = np.column_stack([np.ones(X.shape[0]), X])
	else:
		X = np.column_stack([np.ones(X.shape[0]), X[:,index]])

	theta = np.array([zt for i in range(X.shape[1])])
	otheta = copy.deepcopy(theta)
	for itc in iterLoop(maxIt):
		if batchAlg:
			theta = batch(theta, alfa, X, Y)
		else:
			theta = normal(theta, alfa, X, Y)
		if showplot:
			th.append(([otheta[0],theta[0]], [otheta[1], theta[1]]))
		if(abs(sum(theta-otheta)) < eps):
			break
		otheta = copy.deepcopy(theta)
	if showplot:
		plot(X, Y, theta, analiticna(X,Y), th)
	return theta

def analiticna(X,Y):
	if X.shape[0] != Y.shape[0]:
		X = np.column_stack([np.ones(X.shape[0]), X])
	return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)