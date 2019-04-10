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

#
#
#   Border Cleanup
#
#

def dePad(f,offset):
    return f[offset:f.shape[0]+offset,offset:f.shape[1]+offset]

def padOne(f,offset):
    fp = np.ones((f.shape[0]+2*offset,f.shape[1]+2*offset))
    fp[offset:f.shape[0]+offset,offset:f.shape[1]+offset] = f
    return fp

def cleanupBorder(image,B):
    offset = 1
    padded = padOne(image,offset).astype(np.bool)
    label = measure.label(padded)
    myLabel = label[0,0]
    newImage = np.zeros(padded.shape)
    for (x,y),value in np.ndenumerate(label):
        if value == myLabel or value == 0:
            newImage[x,y] = 0
        else:
            newImage[x,y] = 1
    return dePad(newImage,offset)

particles = readImage("particles.tif").astype(np.bool)
displayImage(particles)

B = morphology.disk(3)

Q_particles = cleanupBorder(particles,B)
displayImage(Q_particles)
saveImage(Q_particles,"Q_particles.png")
