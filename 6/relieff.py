import Orange
data = Orange.data.Table("data/train.tab")

import random
import numpy as np
alfa = 0.05
nrPerm = 100
gain = Orange.feature.scoring.Relief()
ig = np.array([gain(feature, data) for feature in data.domain.features])
suma = np.zeros(len(data.domain.features))
for s in xrange(nrPerm):
	c = [d.get_class() for d in data]
	random.shuffle(c)
	[d.set_class(c[i]) for i,d in enumerate(data)]

	igp = np.array([gain(feature, data) for feature in data.domain.features])
	suma += np.greater(igp, ig) 
suma /= nrPerm
np.save("relieff%d" % (nrPerm), np.nonzero(suma<alfa)[0])