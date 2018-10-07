import numpy as np
import matplotlib.pyplot as plt

#   Returns the context of the file 'name'
def readFile(name):
    f = open(name, 'r')
    contents = f.read()
    f.close()
    return contents


#   Lists all natural and # notes
def notes():
    doremi = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return doremi


#   Lists all extra notes not included in the notes function
def extraNotes():
    doremiExtra = ["Bb","Cb","Db","Eb","Fb","Gb","Ab","E#","B#"]
    return doremiExtra


#   Lists all #, b or natural chords
def chordList():
    chordlist = ["F_d","B_m7","F_m","D_m6","D_m7","D_M4","E_d","E_m","DbM7",
                "F_M","F#M4","E_M","G#d","Dbm7","F#m6","EbM","G#m","Ebd","G#d7",
                "G_m7","G_m6","G#M","Dbd7","A#d","F#d7","D_d7","F_m6","F_m7",
                "D_M6","D_M7","D#d6","B_d7","A#d7","DbM","B_M","F_M6","F_M7",
                "F_M4","A_M","F#M","Dbm","B_m","B_d","Dbd","C#d7","A_d","F#d",
                "B_m6","F#M7","A_M7","A_M6","A_m","A_M4","F#m","C_d6","C_d7",
                "Bbm","Bbd","G_M","D_m","A_m7","A_m6","A_m4","C_M4","C_M7",
                "C_M6","BbM","G_m","D_M","G_d","C#m7","C#m","EbM7","Bbm6",
                "C#d6","C#M7","Abd","Abm","BbM7","C#d","E_m7","E_m6","AbM",
                "C#M","F#m7","C_m7","C_m6","D#M","E_M7","E_M4","D#d7","G_M4",
                "G_M7","G_M6","C_M","F_d7","B_M7","B_M4","D#m","C#M4","D#d",
                "C_m"]
    return chordlist


#   Lists all # or natural chords
def enharmonicChordList():
    enharmonicchordlist  = ["F_d","B_m7","B#m","B#d","F_m","D_m6","D_m7",
                            "D_M4","A#M","E_d","E_m","B#M","F_M","F#M4","E_M",
                            "G#d","F#m6","F#m7","G#m","G#d7","G_m7","G_m6",
                            "G#M","A#d","A#m","F#d7","D_d7","F_m6","F_m7",
                            "D_M6","D_M7","D#d6","B_d7","A#d7","B_M","F_M6",
                            "F_M7","F_M4","A#m6","A_M","F#M","B_m","B_d","C#d7",
                            "A_d","F#d","B_m6","F#M7","A_M7","A_M6","A_m",
                            "A_M4","F#m","C_d6","C_d7","G_M","D_m","A_m7",
                            "A_m6","A_m4","C_M4","C_M7","C_M6","G_m","D_M",
                            "G_d","C#m7","D#M7","A#M7","C#d6","C#M7","C#m",
                            "C#d","E_m7","E_m6","C#M","C_m7","C_m6","D#M",
                            "E_M7","E_M4","D#d7","G_M4","G_M7","G_M6","C_M",
                            "F_d7","B_M7","B_M4","D#m","C#M4","D#d","C_m"]
    return enharmonicchordlist


#   Creates a dictionary with every key as a chord
#   The key's default value is set using the 'default' parameter
#   Can vary depending on whether or not enharmonic
#   equivalents are used, and thus the setting is passed
#   as a parameter
def createChordDict(default,useEnharmonics):
    if useEnharmonics:
        myChordList = enharmonicChordList()
    else:
        myChordList = chordList()
    chordDict = {}
    for i in myChordList:
        chordDict[i] = 0
    return chordDict


#   Creates a dictionary to hold attribute information
#   Can vary depending on whether or not enharmonic
#   equivalents are used, and thus the setting is passed
#   as a parameter
def createChordInfo(useEnharmonics):
    chord = {}
    doremi = notes()
    for i in doremi:
        chord[i] = 0
        chord["b"+i] = 0
    if not useEnharmonics:
        doremiExtra = extraNotes()
        for i in doremiExtra:
            chord["b"+i] = 0
    for i in range(1, 6):
        chord["m"+str(i)] = 0
    return chord


#   Transforms a note into its enharmonic
#   equivalent
def enharmonise(note):
    if note == "Cb":
        enharmonic = "B"
    elif note == "B#":
        enharmonic = "C"
    elif note == "Db":
        enharmonic = "C#"
    elif note == "Eb":
        enharmonic = "D#"
    elif note == "Fb":
        enharmonic = "E"
    elif note == "E#":
        enharmonic = "F"
    elif note == "Gb":
        enharmonic = "F#"
    elif note == "Ab":
        enharmonic = "G#"
    elif note == "Bb":
        enharmonic = "A#"
    else:
        enharmonic = note
    return enharmonic


#   Transforms a chord into its enharmonic
#   equivalent
def chordEnharmonise(chord):
    chord = chord.replace("B#", "C_")
    chord = chord.replace("Cb", "B_")
    chord = chord.replace("Db", "C#")
    chord = chord.replace("Eb", "D#")
    chord = chord.replace("Fb", "E_")
    chord = chord.replace("E#", "F_")
    chord = chord.replace("Gb", "F#")
    chord = chord.replace("Ab", "B#")
    chord = chord.replace("Bb", "A#")
    return chord


#   Creates lists of chords. This is never called
#   and is used only to assist in creating the
#   chordList and enharmonicChordList functions
#   above
def createChordList(useEnharmonics, target):
    rawText = readFile("resources/chord_distribution.txt")
    chordDist = rawText.split("\n")
    chordList = []
    for i in range(len(chordDist)):
        thisLine = chordDist[i].split()
        thisChordColon = thisLine[0]
        thisChord = thisChordColon[:-1]
        if useEnharmonics:
            chordList.append(chordEnharmonise(thisChord))
        else:
            chordList.append(thisChord)
    chordList = list(set(chordList))
    contents = "["
    for i in chordList:
        contents += "\""+i+"\","
    contents = contents[:-1]
    contents += "]"
    f = open(target, 'w')
    f.write(contents)
    f.close()


#   Save confusion results
def saveConfusionCSV(bigConfusion, chordPosition):
    #   Save confusion matrix as CSV
    np.savetxt("confusion.csv", bigConfusion, delimiter=",")
    #   Save chord positions
    contents = ""
    for i in range(len(chordPosition)):
        contents += chordPosition.keys()[chordPosition.values().index(i)] + ","
    contents = contents[:-1]
    print contents
    f = open("chordPos.txt", 'w')
    f.write(contents)
    f.close()


#   Makes an ordered list of chord positions
def makeChordPositionList(chordPosition):
    myList = []
    for i in range(len(chordPosition)):
        myList.append(chordPosition.keys()[chordPosition.values().index(i)])
    return myList


#   Get false positive of a number from the confusion matrix
def getFalsePositive(bigConfusion, i):
    myRow = bigConfusion[i, :]
    fp = 0
    for j in range(len(myRow)):
        if j != i:
            fp += myRow[j]
    return fp


#   Plot confusion matrix as bar graph. This function is a heavily
#   modified version of the code from:
#   https://matplotlib.org/examples/lines_bars_and_markers/barh_demo.html
#
#   This is never actually used because the bar graphs ended
#   up looking pretty terrible
def plotBarGraph(bigConfusion, chordPosition):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    #   Data
    labels = makeChordPositionList(chordPosition)
    y_pos = np.arange((len(labels)))
    truePos = []
    falsePos = []
    for i in range(len(labels)):
        truePos.append(bigConfusion[i][i])
        falsePos.append(getFalsePositive(bigConfusion, i))
    w = 0.3
    ax.barh(y_pos-w, truePos, align='center', color='green')
    ax.barh(y_pos+w, falsePos, align='center', color='red')
    ax.autoscale(tight=True)
    ax.set_yticks(y_pos+w)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('True Positive')
    ax.set_title('True Positive vs. False Positive')

    plt.show()


#   Save the confusion matrix to a text file
#   Takes in the file name, the confusion matrix, the dictionary of chord
#   positions and a boolean that controls whether or not brackets are
#   used to emphasize diagonal elements
def saveConfusion(filename, bigConfusion, chordPosition, useEmphasis):
    chordList = makeChordPositionList(chordPosition)
    contents = ""
    for i in range(len(chordList)):
        contents += chordList[i] + ","
    contents = contents[:-1]
    for i in range(len(chordList)):
        contents += "\n"
        contents += chordList[i] + ","
        for j in range(len(chordList)):
            if i == j and useEmphasis:
                contents += "[" + str(bigConfusion[i][j]) + "],"
            else:
                contents += str(bigConfusion[i][j]) + ","
        contents = contents[:-1]
    f = open(filename, 'w')
    f.write(contents)
    f.close()
