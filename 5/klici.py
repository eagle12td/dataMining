import Orange
import numpy as np
data = Orange.data.Table("housing")
X, Y, _ = data.to_numpy()

m = 3
X /= m
Y /= m

import funkcije as f
print f.getTheta(X, Y, 7, 0.0001, 1e-5, -1, True, True)