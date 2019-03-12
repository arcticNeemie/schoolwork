import sys
import numpy as np

#Writes the numpy array to the file
def writeToFile(arr,file):
    r = arr.shape[0]
    d = arr.shape[1]

    output = ""
    for i in range(r):
        line = ""
        for j in range(d):
            line += str(arr[i][j]) + "\n"
        line = line[:-1]+"\n"
        output += line

    output = output[:-1]
    f = open(file,"w")
    f.write(output)
    f.close()


if(len(sys.argv)!=4):
    #Incorrect usage
    print("Usage: ",sys.argv[0]," <dimension> <number of points in P> <number of points in Q>")
    quit()
else:
    #Initialize
    d = int(sys.argv[1])
    m = int(sys.argv[2])
    n = int(sys.argv[3])

    #Generate Points
    P = np.random.rand(m,d)
    Q = np.random.rand(n,d)

    #Write to file
    writeToFile(P,"p.txt")
    writeToFile(Q,"q.txt")
