import NBControl as nb
import numpy as np
from random import randint
from math import ceil
from math import log
from decimal import *

reviewText = nb.readFile('movie-pang02.csv')        #   Read reviews from file

#   Settings

trainingPercent = 0.9       #Percentage of reviews to be used for training
getcontext().prec = 100     #Precision of decimals

k = 1               #Laplace Smoothing Numerator
nk = 2              #Laplace Smoothing Denominator

bigConfusion = np.zeros([2, 2])
bigIter = 60

for iteration in range(bigIter):
    #   Generate list of reviews

    allReviews = reviewText.split('\n')     #Split text into reviews
    allReviews.pop(0)              #Remove table head
    trainingNo = int(ceil(trainingPercent*len(allReviews)))     #Exact number of reviews for training
    trainingReviews = []
    for i in range(trainingNo):
        j = randint(0, len(allReviews) - 1)
        a = allReviews.pop(j)
        trainingReviews.append(a)

    #   Training

    print 'Training begins on iteration ', iteration+1

    posNo = 0       #Number of positive reviews in training set
    negNo = 0       #Number of negative reviews in training set
    posWords = {}   #Dictionary of positive words
    negWords = {}   #Dictionary of negative words
    for i in range(trainingNo):
        #print 'Training Review #',i+1
        firstSplit = trainingReviews[i].split(',')          #Split the review into Sentiment and Text
        thisReviewWordsSet = set(firstSplit[1].split())     # Split review into a set of unique words
        thisReviewWords = list(thisReviewWordsSet)          # Create a list from the set
        mySentiment = firstSplit[0]                         #Take sentiment
        if mySentiment == 'Pos':
            posNo += 1
        else:
            negNo += 1
        for j in range(len(thisReviewWords)):
            negMod = 0
            posMod = 0
            if mySentiment == 'Pos':
                posMod = 1
            else:
                negMod = 1
            if thisReviewWords[j] in posWords:
                posWords[thisReviewWords[j]] += posMod
            else:
                posWords[thisReviewWords[j]] = posMod
            if thisReviewWords[j] in negWords:
                negWords[thisReviewWords[j]] += negMod
            else:
                negWords[thisReviewWords[j]] = negMod

    #   Testing
    print 'Testing begins on iteration ', iteration+1

    confusion = np.zeros([2, 2])
    posProb = 1.0*posNo/trainingNo
    negProb = 1.0*negNo/trainingNo

    for i in range(len(allReviews)):
        #print 'Testing Review #',i+1
        firstSplit = trainingReviews[i].split(',')          # Split the review into Sentiment and Text
        thisReviewWordsSet = set(firstSplit[1].split())     # Split review into a set of unique words
        thisReviewWords = list(thisReviewWordsSet)          #Create a list from the set
        mySentiment = firstSplit[0]                         # Take sentiment
        runningNeg = 0
        runningPos = 0
        wordCount = 0
        for j in posWords:
            if j in thisReviewWords:
                wordCount += 1
                runningPos += log(posWords[j]+k)-log(posNo+nk)
                runningNeg += log(negWords[j]+k)-log(negNo+nk)
            else:
                runningPos += log(1 - 1.0*(posWords[j]+k)/(posNo+nk))
                runningNeg += log(1 - 1.0*(negWords[j]+k)/(negNo+nk))

        #For this, we ignore words that are in review i but not in the dictionary, as the effect is minimal

        PNegPart = Decimal.exp(Decimal(runningNeg))*Decimal(negProb)
        PPosPart = Decimal.exp(Decimal(runningPos))*Decimal(posProb)
        PNeg = PNegPart/(PNegPart+PPosPart)
        PPos = PPosPart/(PNegPart+PPosPart)
        if PNeg >= PPos:
            prediction = 'Neg'
        else:
            prediction = 'Pos'
        if prediction == mySentiment:
            if prediction == 'Pos':
                confusion[0, 0] += 1
            else:
                confusion[1, 1] += 1
        else:
            if prediction == 'Pos':
                confusion[0, 1] += 1
            else:
                confusion[1, 0] += 1

    print 'Finished with iteration ', iteration+1
    bigConfusion += confusion
    intermediateConfusion = np.divide(bigConfusion, iteration+1)
    print 'Current confusion matrix: '
    print intermediateConfusion

bigConfusion = np.divide(bigConfusion, bigIter)
print 'Final confusion matrix: '
print bigConfusion