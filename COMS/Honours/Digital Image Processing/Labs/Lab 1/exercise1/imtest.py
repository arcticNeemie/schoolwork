import matplotlib.pyplot as plt
import numpy as np
# from scipy import misc

# img.dtype
# img.shape
# plt.gray()
# plt.imshow(img)
# plt.show()
# plt.axis("off")


def getImageMean(image):
    return image.mean()


def getImageVariance(image):
    sum = 0
    totalPixels = image.shape[0]*image.shape[1]
    mean = getImageMean(image)
    for (x,y),value in np.ndenumerate(image):
        sum += (value-mean)**2
    return sum/totalPixels


def getFullDynamicRange(image):
    type = image.dtype
    if np.issubdtype(type, np.integer):
        N = image.itemsize*8
        return 2**N - 1
    else:
        return 1


def reverseImage(image):
    max = getFullDynamicRange(image)
    image = max-image
    return image


def displayImage(image):
    max = getFullDynamicRange(image)
    plt.gray()
    plt.imshow(image, vmin=0, vmax=max)
    plt.show()
    plt.axis("off")


def contrastStretch(image,plow,phigh):
    newImage = image.copy()
    max = getFullDynamicRange(image)
    for (x,y),value in np.ndenumerate(image):
        if value<phigh and value>plow:
            newImage[x,y] = max*(image[x,y]-plow)/(phigh-plow)
        elif value<plow:
            newImage[x,y] = 0
        else:
            newImage[x,y] = 1
    return newImage


img = plt.imread('images/pollen.png')
img = img.astype(float)
max = getFullDynamicRange(img)
myPValue = 0.3
plow = 0+myPValue
phigh = max-myPValue
img = contrastStretch(img,plow,phigh)
displayImage(img)
