from collections import Counter

files = [open("knn.txt"), open("knnm.txt"), open("RF200por.txt"), open("test.txt")]

a = []
for f in files:
	a.append([[int(j) for j in i.strip().split(" ")] for i in f.readlines()])

b = []
for i in xrange(len(a[0])):
	c = Counter(a[0][i]+a[1][i]+a[2][i])
	x = [x[0] for x in Counter.most_common(c) if x[1]>2]
	if len(x)==0 :
		x = [40]
	b.append(x)

print "done"
f = file("skupniNoTest.csv","w")
f.write("\n".join(   [" ".join([str(x) for x in i]) for i in b ]    ))
f.close()
