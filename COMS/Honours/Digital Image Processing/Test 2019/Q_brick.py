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
#   DFT Filtering
#
#
def DFTFilter(f,h):
    A = f.shape[0]
    B = f.shape[1]
    C = h.shape[0]
    D = h.shape[1]
    P = nextPow2(A+C-1)
    Q = nextPow2(B+D-1)
    fp = np.zeros((P,Q))
    hp = np.zeros((P,Q))
    fp[0:A,0:B] = f
    hp[0:C,0:D] = h

    F = sp.fft2(fp)
    H = sp.fft2(hp)
    G = H*F
    g = sp.ifft2(G)
    return np.real(g[0:A,0:B])


def nextPow2(x):
    p = 1
    while 2**p < x:
        p+=1
    return 2**p

brick = readImage("brick.png")
Sobel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
displayImage(brick)
Q_brick = DFTFilter(brick,Sobel)
displayImage(Q_brick)
saveImage(Q_brick,"Q_brick.png")
