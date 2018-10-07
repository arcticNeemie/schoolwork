import numpy as np
from matplotlib import pyplot as plt

points = []

filename = "reg_0" #File containing relevant points

f = open(filename + ".txt", "r")
for line in f:
	point = line.split(",")
	points.append([float(point[0]), 100*float(point[1])])

p = np.asarray(points)

x, y = p.T

fig = plt.figure()
plt.scatter(x, y)
fig.suptitle('Test Data Accuracy (' + filename + ')', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
fig.savefig(filename + '.jpg')

