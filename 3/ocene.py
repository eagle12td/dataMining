import Orange

def cvkf(learner, data, k):
	n = len(data)
	selection = [int(i/(float(n)/k)) for i in xrange(n)]
	res = []
	for i in xrange(k):
		cl = learner(data.select(selection, i, negate=1))
		res += [cl(t) for t in data.select(selection, i)]
	return res

# returns precision between two lists
# t = true list (list of all correct classes)
# p = predicted list (list of our predictions)
def precision(t,p):
	return len(set(t).intersection(set(p)))/len(set(p))

def recall(t,p):
	return len(set(t).intersection(set(p)))/len(set(t))

def fscore(t,p):
	pp = precision(t,p)
	rr = recall(t,p)
	return 2*pp*rr/(pp+rr)

def avgfscore(t,p):
	return sum([fscore(t[i], p[i]) for i in range(len(t))])/len(t)
