import NBControl as nb
import numpy as np
from random import randint
from math import exp

reviewText = nb.readFile('simple-food-reviews.txt')  #Read reviews from file


trainingNo = 18

reviews = reviewText.split('\n')        #Split text into reviews

#Do training

words = []
positiveCount = []
negativeCount = []
positiveNo = 0
negativeNo = 0
totalNo = trainingNo

for i in range(trainingNo):
    thisReviewWords = reviews[i].split()        #Split review into its words
    thisSentiment = thisReviewWords[0]                  #Take sentiment
    if thisSentiment == "-1":
        negativeNo = negativeNo + 1
    else:
        positiveNo = positiveNo + 1
    for j in range(1,len(thisReviewWords)):
        if len(thisReviewWords[j]) > 2:
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
stillOn = True
while stillOn:
    newReview = raw_input("Enter a review: ")
    if(newReview == '-1'):
        print 'Bye'
        break
    thisReviewWords = newReview.split()    #Split review into words
    runningProbPos = 1
    runningProbNeg = 1
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
    negProb = 1.0 * (runningProbNeg * PNeg) / ((runningProbNeg * PNeg) + (runningProbPos * PPos))
    posProb = 1.0 * (runningProbPos * PPos) / ((runningProbNeg * PNeg) + (runningProbPos * PPos))
    if negProb >= posProb:
        prediction = 'negative'
    else:
        prediction = 'positive'
    print 'I am ',int(round(100.0*max([negProb,posProb]))),'% sure that this review is ',prediction