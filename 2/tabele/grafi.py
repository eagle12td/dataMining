import pylab as p
import numpy as np
import calendar

allC = np.load("class.npy")
allN = []

table100_000 = {}
table100_005 = {}
table500_000 = {}
table500_005 = {}
for i in range(100):
	if "c"+str(i) in allC:
		allN.append(i)
		table100_000[i] = (np.load("100_000/c"+str(i)+".npy")[0])
		table100_005[i] = (np.load("100_005/c"+str(i)+".npy")[0])
		table500_000[i] = (np.load("500_000/c"+str(i)+".npy")[0])
		table500_005[i] = (np.load("500_005/c"+str(i)+".npy")[0])

histData500_000 = []
histData100_000 = []
for i in allN:
	histData500_000.append(len(table500_000[i]))
	histData100_000.append(len(table100_000[i]))

p.close()
p.xlim([0, 84])
p1 = p.bar(allN, histData100_000, .7, color='b', align="center")
p2 = p.bar(allN, histData500_000, .7, color='r', align="center")
p.xlabel("Razred")
p.ylabel("Stevilo atributov")
p.title("St. atributov za posamezni razred pri: alfa = 0.00")
p.legend([p1, p2], ["100x random", "500x random"])
p.savefig("1.png")

histData500_005 = []
histData100_005 = []
for i in allN:
	histData500_005.append(len(table500_005[i]))
	histData100_005.append(len(table100_005[i]))

p.close()
p.xlim([0, 84])
p1 = p.bar(allN, histData100_005, .7, color='b', align="center")
p2 = p.bar(allN, histData500_005, .7, color='r', align="center")
p.xlabel("Razred")
p.ylabel("Stevilo atributov")
p.title("St. atributov za posamezni razred pri: alfa = 0.05")
p.legend([p1, p2], ["100x random", "500x random"])
p.savefig("2.png")

p.close()
p.xlim([0, 2025])
p.ylim([0, 83])
p.xlabel("Atribut")
p.ylabel("Razred")
p.title("Uporabni atributi za razrede: alfa = 0.00, 100x random")
for i,d in table100_000.items():
	for j in d:
		p.plot(j, i, 'b,')
p.savefig("3.png")

p.close()
p.xlim([0, 2025])
p.ylim([0, 83])
p.xlabel("Atribut")
p.ylabel("Razred")
p.title("Uporabni atributi za razrede: alfa = 0.05, 100x random")
for i,d in table100_005.items():
	for j in d:
		p.plot(j, i, 'b,')
p.savefig("4.png")

p.close()
p.xlim([0, 2025])
p.ylim([0, 83])
p.xlabel("Atribut")
p.ylabel("Razred")
p.title("Uporabni atributi za razrede: alfa = 0.00, 500x random")
for i,d in table500_000.items():
	for j in d:
		p.plot(j, i, 'b,')
p.savefig("5.png")

p.close()
p.xlim([0, 2025])
p.ylim([0, 83])
p.xlabel("Atribut")
p.ylabel("Razred")
p.title("Uporabni atributi za razrede: alfa = 0.05, 500x random")
for i,d in table500_005.items():
	for j in d:
		p.plot(j, i, 'b,')
p.savefig("6.png")