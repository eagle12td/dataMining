import Orange
import numpy as np
data = Orange.data.Table("housing")
X, Y, _ = data.to_numpy()

m = 3
X /= m
Y /= m

import funkcije as f
print f.getTheta(X, Y, 0.0001, 1e-5, 7, -1, True, False) #izris

rez,it = f.getTheta(X, Y, 0.000001, 1e-5, batchAlg=False)
print rez,it
#rez = np.array([3.40771294, -0.135737901, 0.0357196662, 0.0891750653, 0.755742642, -13.8276397, 5.31945778, 0.00785609675, -1.05219413, 0.136973014, -0.00759340766, -0.441920824, 0.00957698475, -0.465608380])
#iteracij: 1001628
print f.j(rez, X, Y)

rez,it = f.getTheta(X, Y, 0.0000001, 1e-5, batchAlg=True)
print rez,it
#rez = np.array([-8.65614691, -0.119746420, 0.0568844863, 0.250145010, -10.28511096, -10.78191477, 8.72749238, 0.0397324835, -0.762339775, 0.184630329, -0.00996154857, -0.0796754944, 0.0250695631, -0.296737765]) 
#iteracij: 940849
print f.j(rez, X, Y)

rez = f.analiticna(np.column_stack([np.ones(X.shape[0]), X]), Y)
print f.j(rez, X, Y)