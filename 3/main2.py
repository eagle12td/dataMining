import Orange
import cPickle

#learners = [
#    Orange.multilabel.BinaryRelevanceLearner(name="br"),
    #Orange.multilabel.BinaryRelevanceLearner(name="br", base_learner=Orange.classification.knn.kNNLearner),
    #Orange.multilabel.LabelPowersetLearner(name="lp")#,!!!0.299731357032
    #Orange.multilabel.LabelPowersetLearner(name="lp", base_learner=Orange.classification.knn.kNNLearner),!!!0.299731357032
    #Orange.multilabel.MLkNNLearner(name="mlknn",k=5),0.153137263994
    #Orange.multilabel.BRkNNLearner(name="brknn",k=5),0.218061763966
#]

#data = Orange.data.Table("train")

#learner = Orange.multilabel.LabelPowersetLearner()
#classifier = learner(data)
#cPickle.dump(classifier, file("minidata/trainClass.pickled", "w"), -1)
trainClass = cPickle.load(file("minidata/trainClass.pickled"))
test = cPickle.load(file("minidata/testDataO.pickled"))
for i in test:
	a = trainClass(i)
	t = sorted([int(j.variable.name[1:]) for j in a if j==1])
	print t



#res = Orange.evaluation.testing.cross_validation(learners, data,2)
#loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
#accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
#precision = Orange.evaluation.scoring.mlc_precision(res)
#recall = Orange.evaluation.scoring.mlc_recall(res)
#print 'loss=', loss
#print 'accuracy=', accuracy
#print 'precision=', precision
#print 'recall=', recall
#print 'F=', 2*(recall[0]*precision[0]/(recall[0]+precision[0]))