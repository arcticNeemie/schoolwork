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
#   Noise Removal
#
#

def removeSaltAndPepper(image,B):
    im_open = morphology.binary_opening(image,B)
    im_close = morphology.binary_closing(im_open,B)
    return im_close

square_noisy = readImage("square_noisy.png").astype(np.bool)
displayImage(square_noisy)

B = morphology.disk(13)

Q_square = removeSaltAndPepper(square_noisy,B)
displayImage(Q_square)
saveImage(Q_square,"Q_square.png")
