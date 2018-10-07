import NBControl as nb
import numpy as np
from random import randint
from math import exp

reviewText = nb.readFile('simple-food-reviews.txt')  #Read reviews from file


trainingNo = 12
bigConfusion = np.zeros([2,2])
bigIteration = 1000
for iteration in range(bigIteration):
    reviews = reviewText.split('\n')        #Split text into reviews

    #Generate a random list of 12 reviews for training
    trainingReviews = []
    for i in range(trainingNo):
        j = randint(0,len(reviews)-1)
        a = reviews.pop(j)
        trainingReviews.append(a)

    #Do training

    words = []
    positiveCount = []
    negativeCount = []
    positiveNo = 0
    negativeNo = 0
    totalNo = trainingNo

    for i in range(trainingNo):
        thisReviewWords = trainingReviews[i].split()        #Split review into its words
        thisSentiment = thisReviewWords[0]                  #Take sentiment
        if thisSentiment == "-1":
            negativeNo = negativeNo + 1
        else:
            positiveNo = positiveNo + 1
        for j in range(1,len(thisReviewWords)):
            if thisReviewWords[j] in words:
                index = words.index(thisReviewWords[j])
                if thisSentiment == "-1":
                    negativeCount[index] = negativeCount[index] + 1
                else:
                    positiveCount[index] = positiveCount[index] + 1
            else:
                words.append(thisReviewWords[j])
                if thisSentiment == "-1":
                    negativeCount.append(1)
                    positiveCount.append(0)
                else:
                    positiveCount.append(1)
                    negativeCount.append(0)


    #Do testing

    PNeg = 1.0*negativeNo/totalNo
    PPos = 1.0*positiveNo/totalNo
    totalCount = np.array(negativeCount)+np.array(positiveCount)
    k = 1
    nk = 2
    confusion = np.zeros([2, 2])
    for i in range(len(reviews)):
        thisReviewWords = reviews[i].split()    #Split review into words
        mySentiment = thisReviewWords.pop(0)
        runningProbPos = 1.0
        runningProbNeg = 1.0
        for j in range(len(thisReviewWords)):
            if thisReviewWords[j] not in words:
                runningProbNeg *= 1.0 * k / (negativeNo + nk)
                runningProbPos *= 1.0 * k / (positiveNo + nk)
        for j in range(len(words)):
            if words[j] in thisReviewWords:
                if negativeCount[j]==0:
                    runningProbNeg *= 1.0 * k / (negativeNo + nk)
                else:
                    runningProbNeg *= 1.0 * (negativeCount[j]) / (negativeNo)
                if positiveCount[j]==0:
                    runningProbPos *= 1.0 * k / (positiveNo + nk)
                else:
                    runningProbPos *= 1.0 * (positiveCount[j]) / (positiveNo)
            else:
                runningProbNeg *= 1.0 * (1 - (negativeCount[j] + k) / (negativeNo + nk))
                runningProbPos *= 1.0 * (1 - (positiveCount[j] + k) / (positiveNo + nk))
        if runningProbPos == 0.0 and runningProbNeg == 0.0:
            continue
        negProb = 1.0 * (runningProbNeg * PNeg) / ((runningProbNeg * PNeg) + (runningProbPos * PPos))
        posProb = 1.0 * (runningProbPos * PPos) / ((runningProbNeg * PNeg) + (runningProbPos * PPos))
        if negProb >= posProb:
            prediction = '-1'
        else:
            prediction = '1'
        if prediction == mySentiment:
            if prediction == '1':
                confusion[0,0] += 1
            else:
                confusion[1,1] += 1
        else:
            if prediction == '1':
                confusion[0,1] += 1
            else:
                confusion[1,0] += 1
    bigConfusion += confusion

bigConfusion = np.divide(bigConfusion,bigIteration)
print(bigConfusion)