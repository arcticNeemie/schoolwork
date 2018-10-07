#This is supposed to contain a bunch of commonly used functions for the questions 

def readFile(name):
    f = open(name,'r')
    contents = f.read()
    f.close()
    return contents
