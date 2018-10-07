import BachControl as BC
import numpy as np
from math import ceil
from math import log
from random import randint
from copy import copy
from decimal import *

############################### Naive Bach ###############################
#
#   This program classifies events in Bach chorals based on present notes,
#   the bass note and the accent (or meter) of the event. It does so using
#   Naive-Bayes classification with Laplace smoothing. The dataset itself is
#   described in detail in the jsbach_chorals_harmony.names file in the
#   resources folder.
#

choralEvents = BC.readFile("resources/jsbach_chorals_harmony.data")     # Read in data from file

#   Note - Position in Data Array
#   C       = 3
#   C#      = 4
#   D       = 5
#   D#      = 6
#   E       = 7
#   F       = 8
#   F#      = 9
#   G       = 10
#   G#      = 11
#   A       = 12
#   A#      = 13
#   B       = 14


#   Settings
trainingPercent = 0.8   # Percentage of data used to train
k = 1       # Laplace Smoothing
nk = 2
bigIter = 50    # Number of times the learning and testing is done
#   Feature Control
useChordNotes = True    # Controls whether or not chord notes are a feature
useBassNotes = True     # Controls whether or not bass notes are a feature
useMeter = False         # Controls whether or not meter is a feature
#   Use Enharmonic Equivalents for classes
useEnharmonics = True

#   Position of each chord in the confusion matrix
chordPosition = BC.createChordDict(0, useEnharmonics)
i = 0
for chord in chordPosition:
    chordPosition[chord] = i
    i += 1

#   Average confusion matrix
bigConfusion = np.zeros([len(chordPosition), len(chordPosition)])

#   Split Data into Training and Testing
allTheData = choralEvents.split("\n")                      # Split into events
trainingNo = int(ceil(trainingPercent*len(allTheData)))    # Get number of training events
testingNo = len(allTheData)-trainingNo                     # Get number of testing events

doremi = BC.notes()     # List of Notes

for iteration in range(bigIter):
    #   Randomly split into Training and Testing sets
    allData = copy(allTheData)    # Reset to full dataset
    trainingData = []
    for i in range(trainingNo):             # Generate list of training events
        j = randint(0, len(allData)-1)
        trainingEvent = allData.pop(j)
        trainingData.append(trainingEvent)

    #   Training
    chordNums = BC.createChordDict(0, useEnharmonics)            # Tracks number of each chord
    chordDicts = BC.createChordDict({}, useEnharmonics)         # Tracks numbers of each attribute of each chord
    for i in chordDicts:
        chordDicts[i] = BC.createChordInfo(useEnharmonics)

    for i in range(trainingNo):
        print "Training event ", i+1, " on iteration ", iteration+1
        myEventString = trainingData[i]
        myEvent = myEventString.split(",")
        myChord = myEvent[16]
        if useEnharmonics:
            myChord = BC.chordEnharmonise(myChord[1:])
        else:
            myChord = myChord[1:]
        chordNums[myChord] += 1     # Add to chord number
        #   Chord Notes
        if useChordNotes:
            for j in range(12):
                if myEvent[j+2] == "YES":
                    chordDicts[myChord][doremi[j]] += 1
        # Bass Note
        if useBassNotes:
            if useEnharmonics:
                chordDicts[myChord]["b"+BC.enharmonise(myEvent[14])] += 1
            else:
                chordDicts[myChord]["b"+myEvent[14]] += 1
        # Meter
        if useMeter:
            chordDicts[myChord]["m"+str(myEvent[15])] += 1

    #   Testing
    confusion = np.zeros([len(chordNums), len(chordNums)])
    chordProbs = BC.createChordDict(0, useEnharmonics)          # Priors

    for i in chordProbs:
        chordProbs[i] = 1.0*chordNums[i]/trainingNo     # Priors for each chord

    trainingEventNo = 1
    for myEventString in allData:
        print "Testing event ", trainingEventNo, " on iteration ", iteration+1
        myEvent = myEventString.split(",")
        myChord = myEvent[16]
        if useEnharmonics:
            myChord = BC.chordEnharmonise(myChord[1:])
        else:
            myChord = myChord[1:]
        myChordProbs = BC.createChordDict(0, useEnharmonics)
        for chord in myChordProbs:
            running = 0
            # Chord Notes
            if useChordNotes:
                for j in range(12):
                    if myEvent[j+2] == "YES":
                        running += log(chordDicts[chord][doremi[j]]+k) - log(chordNums[chord]+nk)
                    else:
                        running += log(1 - (chordDicts[chord][doremi[j]]+k)/(chordNums[chord]+nk))
            # Bass Note
            if useBassNotes:
                for note in doremi:
                    if myEvent[14] == note:
                        running += log(chordDicts[chord]["b"+note]+k) - log(chordNums[chord]+nk)
                    else:
                        running += log(1 - (chordDicts[chord]["b"+note] + k) / (chordNums[chord] + nk))
            # Meter
            if useMeter:
                for i in range(1, 6):
                    if myEvent[15] == str(i):
                        running += log(chordDicts[chord]["m"+str(i)] + k) - log(chordNums[chord] + nk)
                    else:
                        running += log(1 - (chordDicts[chord]["m"+str(i)] + k) / (chordNums[chord] + nk))
            myChordProbs[chord] = running
        #   Probability Calculations
        chordProbPart = BC.createChordDict(0, useEnharmonics)       # Contains part of the Naive Bayes formula
        chordProbPartSum = 0
        for chord in chordProbPart:
            thisChordRunningProb = Decimal.exp(Decimal(myChordProbs[chord]))*Decimal(chordProbs[chord])
            chordProbPart[chord] = thisChordRunningProb
            chordProbPartSum += thisChordRunningProb    # Sum the probabilities together
        myEventChordProb = BC.createChordDict(0, useEnharmonics)    # Contains the entire Naive Bayes formula
        maxProb = 0
        currentChord = ""
        for chord in myEventChordProb:
            thisEventChordProb = chordProbPart[chord]/chordProbPartSum      # Probability for class/sum of probabilities
            myEventChordProb[chord] = thisEventChordProb
            if thisEventChordProb > maxProb:        # Pick one with highest probability
                maxProb = thisEventChordProb
                currentChord = chord
        # Update Confusion Matrix
        confusion[chordPosition[myChord]][chordPosition[currentChord]] += 1
        trainingEventNo += 1
    # Update Big Confusion
    bigConfusion += confusion

#   Average Confusion Matrix
bigConfusion = np.divide(bigConfusion, bigIter)
accuracy = 0
for i in chordPosition:
    accuracy += bigConfusion[chordPosition[i]][chordPosition[i]]
accuracy = 1.0*accuracy/testingNo
print accuracy

#   Save confusion matrix to file
filename1 = "Report/Confusion/NB_confusion_5.txt"
filename2 = "Report/Confusion/NB_confusion_5_emph.txt"
BC.saveConfusion(filename1, bigConfusion, chordPosition, False)
BC.saveConfusion(filename2, bigConfusion, chordPosition, True)