import NBControl as nb
import numpy as np
from random import randint
from math import ceil
from math import log
from decimal import *

digitText = nb.readFile('smalldigits.csv')        # Read digits from file

#   Settings

trainingPercent = 0.8       # Percentage of digits to be used for training
getcontext().prec = 100     # Precision of decimals

k = 1               # Laplace Smoothing Numerator
nk = 2              # Laplace Smoothing Denominator

bigConfusion = np.zeros([10, 10])
bigIter = 100

for iteration in range(bigIter):
    # Generate list of digits
    allDigits = digitText.split('\n')  # Split text into each digit
    trainingNo = int(ceil(trainingPercent * len(allDigits)))  # Exact number of digits for training
    trainingDigits = []
    for i in range(trainingNo):
        j = randint(0, len(allDigits) - 1)
        a = allDigits.pop(j)
        trainingDigits.append(a)

    # Training
    print 'Training begins on iteration #',iteration
    digitNo = np.zeros(10)          # List of number of occurrence of each digit
    pixelMaps = []                  # Create a list of arrays, each one of which will list occurences of pixels for a given digit
    for i in range(10):
        pixelMaps.append(np.zeros(64))

    for i in range(trainingNo):
        # print 'Training digit ',i+1
        thisDigit = trainingDigits[i].split(',')        # Split digit into individual pixels (64) + digit
        myDigit = int(thisDigit.pop(64))                # Get actual digit
        thisDigit = np.asarray(np.array(thisDigit),int)
        digitNo[myDigit] += 1
        pixelMaps[myDigit] += thisDigit                 # Add this digit's array to the count array

    # Testing
    print 'Testing begins on iteration #',iteration

    confusion = np.zeros([10,10])
    digitProbs = []
    for i in range(10):
        digitProbs.append(1.0*digitNo[i]/trainingNo)

    for i in range(len(allDigits)):
        # print 'Testing digit ',i+1
        thisDigit = trainingDigits[i].split(',')            # Split digit into individual pixels (64) + digit
        myDigit = int(thisDigit.pop(64))                    # Get actual digit
        runnings = np.zeros(10)
        for j in range(len(thisDigit)):
            if thisDigit[j] == '1':
                for m in range(10):
                    runnings[m] += log(pixelMaps[m][j]+k)-log(digitNo[m]+nk)
            else:
                for m in range(10):
                    runnings[m] += log(1 - 1.0*(pixelMaps[m][j]+k)/(digitNo[m]+nk))

        probPart = []
        denom = 0
        for j in range(10):
            number = Decimal.exp(Decimal(runnings[j]))*Decimal(digitProbs[j])
            probPart.append(number)
            denom += number

        myProb = []
        for j in range(10):
            myProb.append(probPart[j]/denom)

        prediction = myProb.index(max(myProb))
        confusion[prediction,myDigit] += 1

    print 'Finished with iteration ',iteration+1
    bigConfusion += confusion
    intermediateConfusion = np.divide(bigConfusion, iteration + 1)
    print 'Current confusion matrix: '
    print intermediateConfusion
    accuracy = 0
    for j in range(10):
        accuracy += intermediateConfusion[j, j]
    accuracy = 1.0 * accuracy / len(allDigits)
    print 'Current Accuracy = ', 100 * accuracy, '%'


bigConfusion = np.divide(bigConfusion, bigIter)
print 'Final confusion matrix: '
print bigConfusion
accuracy = 0
for j in range(10):
    accuracy += bigConfusion[j, j]
accuracy = 1.0*accuracy/len(allDigits)
print 'Accuracy = ', 100*accuracy, '%'
