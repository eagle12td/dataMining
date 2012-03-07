from collections import Counter
f = open("minidata/trainingLabels.csv")
a = []
for line in f:
    a.append([int(v) for v in line.strip().split(",")])
#cc = Counter([i for i in a])
cc = Counter()
for i in a:
	cc.update(i)
ccc = [i[0] for i in cc.most_common(25)]
c = Counter([len(i) for i in a])

f = open("tt.txt")
fout = file("tt.txt.reduced", "w")
a = []
for line in f:
    temp = [i for i in [int(v) for v in line.strip().split(" ")] if i in ccc]
    for i in temp:
        fout.write(str(i)+" ")
    if temp == []:
        fout.write("40")
    fout.write("\n")
fout.close()
    
#mostCommon = [i[0] for i in dd.most_common(4)]