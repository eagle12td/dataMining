import matplotlib.pyplot as plt
import numpy as np

matrikaClass = [line.split(',') for line in filter(None, open("trainingLabels.csv").read().strip().split('\n'))]
matrika = [line.split('\t') for line in filter(None, open("trainingData.csv").read().strip().split('\n'))]

stPrimerov = len(matrika)
stAtributov = len(matrika[0])

delezRazlicenOdNic = 1.*((stPrimerov*stAtributov) - sum(i.count('0') for i in matrika))/(stPrimerov*stAtributov)

statPrimeri = [stAtributov - v.count('0') for v in matrika]

statAttr = [stPrimerov - v.count('0') for v in zip(*matrika)]

stRazredov = len(set(reduce(lambda x,y: x+y, matrikaClass)))
statRazredi = [len(set(i)) for i in matrikaClass]

stUporabnih = 0
for i in matrika:
    stevec = 0
    for j in i:
        if stevec >= 30:
            stUporabnih+=1
            break
        if j != '0':
            stevec+=1

print 'Koliko primerov vsebujejo podatki? ODGOVOR: %d' % stPrimerov
print 'Koliko atributov vsebujejo podatki? ODGOVOR: %d' % stAtributov
print 'Kaksnega tipa so atributi? ODGOVOR: zvezni'
print 'Kako redka je matrika oz. kaksen delez njenih elementov ima vrednost razlicno od 0? ODGOVOR: %f%%' % (delezRazlicenOdNic*100)
print 'Koliko je vseh razlicnih oznak (razredov) v podatkih? ODGOVOR: %d' % stRazredov
print 'Koliko atributov, ki so razlicni od 0 ima primer, ki ima najvec atributov enakih 0? ODGOVOR: %d' % min(statPrimeri)
print 'Koliko atributov, ki so razlicni od 0 ima primer, ki ima najvec atributov razlicnih od 0? ODGOVOR: %d' % max(statPrimeri)
print 'Recimo, da so primeri, ki imajo manj kot 30 nenicelnih atributov neuporabni. Koliko je potem uporabnih? ODGOVOR: %d' % stUporabnih
print 'Koliko je atributov, ki imajo povsod vrednost enako 0? ODGOVOR: %d' % statAttr.count(0)

plt.close()
plt.hist(statPrimeri)
plt.ylabel('Stevilo primerov')
plt.xlabel('Stevilo nenicelnih atributov')
plt.savefig('statPrimeri.png')

plt.close()
plt.hist(statAttr, 100)
plt.ylabel('Stevilo atributov')
plt.xlabel('Stevilo primerov za nenicelne atribute')
plt.savefig('statAttr.png')

plt.close()
plt.hist(statRazredi)
plt.xlabel('Stevilo oznak')
plt.ylabel('Stevilo primerov')
plt.savefig('statRazredi.png')