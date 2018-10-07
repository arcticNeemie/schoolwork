import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
n = 100
m = 1
c = 0
noiseMin = -0.5
noiseMax = 0.5

x = np.linspace(-1,1,n)
noise = np.random.uniform(noiseMin,noiseMax,n)
y = m*x + c
y = y+noise



#noisyData = [[1,2,3,4],[5,6,7,8]]
plt.plot(x,m*x+c,"red")
ax.scatter(x, y)
plt.show()

t1, t2, r_value, p_value, std_err = stats.linregress(x,y)


fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
plt.plot(x,m*x+c,"red")
plt.plot(x,t1*x+t2,"green")
ax1.scatter(x, y)
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
plt.plot(x,m*x+c,"red")
plt.plot(x,t1*x+t2,"green")
plt.plot(x,(t1*2)*x+t2,"blue")
plt.plot(x,(t1*-2)*x+t2,"orange")
plt.plot(x,(t1*3)*x+t2,"purple")
plt.plot(x,(t1*-3)*x+t2,"cyan")
ax2.scatter(x, y)
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
plt.plot(x,m*x+c,"red")
plt.plot(x,t1*x+t2,"green")
plt.plot(x,t1*x+(t2+1),"blue")
plt.plot(x,t1*x+(t2+2),"orange")
plt.plot(x,t1*x+(t2-1),"purple")
plt.plot(x,t1*x+(t2-2),"cyan")
ax3.scatter(x, y)
plt.show()
