import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from skimage.color import rgb2gray
from skimage import morphology
from skimage import measure
import copy
import math
import scipy.fftpack as sp

filepath = "test_images/"

def readImage(name):
    return plt.imread(filepath+name)

def displayImage(image,m=1):
    plt.gray()
    plt.imshow(image,vmin=0,vmax=m)
    plt.axis("off")
    plt.show()

def getImageMax(image):
    type = image.dtype
    if np.issubdtype(type,np.integer):
        return np.iinfo(type).max
    else:
        return 1

def saveImage(image,name):
    plt.imsave(name,image,vmin=0,vmax=getImageMax(image))

def plotHist(histogram):
    x = np.arange(len(histogram))
    plt.bar(x,histogram)
    plt.show()

def contrastStretch(image,pl,ph,m=255):
    newImage = np.zeros(image.shape).astype(image.dtype)
    for (x,y),value in np.ndenumerate(image):
        newImage[x,y] = piecewise(value,pl,ph,m)
        #break
    return newImage

def piecewise(value,pl,ph,m):
    if value<=pl:
        return 0
    elif value>pl and value<ph:
        return m*(value-pl)/(ph-pl)
    else:
        return m

#Only for ints
def getHistogram(image):
    m = getImageMax(image)
    histogram = np.zeros(m+1)
    for (x,y),value in np.ndenumerate(image):
        histogram[value]+=1
    return histogram

def getProbs(histogram):
    n = np.sum(histogram)
    probs = []
    for i in range(len(histogram)):
        probs.append(histogram[i]/n)
    return probs

def getCumulative(probs):
    return np.cumsum(probs)

def histogramEqualization(image):
    m = getImageMax(image)
    h = getHistogram(image)
    p = getProbs(h)
    c = getCumulative(p)
    newImage = np.zeros(image.shape).astype(image.dtype)
    for (x,y),value in np.ndenumerate(image):
        newImage[x,y] = m*c[value]
    return newImage

einstein = readImage("einstein.tif")
displayImage(einstein,255)
original_hist = getHistogram(einstein)
plotHist(original_hist)

saveImage(Q_einstein_hist,"Q_einstein_hist.png")

Q_einstein_contrast = contrastStretch(einstein,60,160)
displayImage(Q_einstein_contrast,m=255)
saveImage(Q_einstein_contrast,"Q_einstein_contrast.png")

contrast_hist = getHistogram(Q_einstein_contrast)
plotHist(contrast_hist)

print("Histogram\n\n")
displayImage(einstein,m=255)
plotHist(getHistogram(einstein))

Q_einstein_hist = histogramEqualization(einstein)
displayImage(Q_einstein_hist,m=255)
saveImage(Q_einstein_hist,"Q_einstein_hist.png")
equalized_hist = getHistogram(Q_einstein_hist)
plotHist(equalized_hist)
