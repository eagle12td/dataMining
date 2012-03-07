from collections import defaultdict
import Orange
import jrs
reload(jrs)

mld = jrs.Data()
#learner = Orange.classification.bayes.NaiveLearner()
maj = Orange.classification.majority.MajorityLearner()
#learner = Orange.classification.svm.LinearSVMLearner()

rez = {}
#for cn in mld.classes:
cn = "c40"
data = mld.get_single_class_data(label=cn)
#res = Orange.evaluation.testing.cross_validation([maj, learner], data)
#print Orange.evaluation.scoring.CA(res)
cl = maj(data)
res = Orange.evaluation.testing.cross_validation([maj, learner], data)
	#rez[cn] = [cl(i, Orange.classification.Classifier.GetProbabilities) for i in data]
t = [cl(i) for i in data]
rez[cn] = [i for i in t if i=="T"]
#tab = defaultdict(list)
#for i,d in rez.items():
#	for ii,pp in enumerate(d):
#		if pp["T"] > .9999:
#			tab[ii].append(i)
			
#[tab[ii].append(i) for ii,pp in enumerate(d) if pp["T"] > .9999]