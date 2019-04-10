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
#   Spatial Filtering
#
#

koala = readImage("koala.png")
koala = rgb2gray(koala)
displayImage(koala)

hb = 3*np.array([[0,0,0],[0,1,0],[0,0,0]]) - (2.0/9)*np.ones((3,3))

def applyFilter(f,h):
    offset = math.floor(h.shape[0]/2)
    fp = padZero(f,offset)
    newImage = np.zeros(fp.shape)
    for x in range(offset,fp.shape[0]+offset-1):
        for y in range(offset,f.shape[1]+offset-1):
            g = 0
            for s in range(h.shape[0]):
                for t in range(h.shape[1]):
                    w = h[s,t]
                    F = fp[x-s,y-t]
                    g += w*F
            newImage[x,y] = g
    return dePad(newImage,offset)

def padZero(f,offset):
    fp = np.zeros((f.shape[0]+2*offset,f.shape[1]+2*offset))
    fp[offset:f.shape[0]+offset,offset:f.shape[1]+offset] = f
    return fp

def dePad(f,offset):
    return f[offset:f.shape[0]+offset,offset:f.shape[1]+offset]

Q_koala = applyFilter(koala,hb)
displayImage(Q_koala)
saveImage(Q_koala,"Q_koala.png")
