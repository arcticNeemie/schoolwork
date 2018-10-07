import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
n = 100
a = 1
b = 1
noiseMin = -0.5
noiseMax = 0.5

x = np.linspace(-1,1,n)
noise = np.random.uniform(noiseMin,noiseMax,n)
y = a*np.exp(b*x)
y = y+noise
y = np.absolute(y)

m, c, r_value, p_value, std_err = stats.linregress(x,np.log(y))

t2 = m
t1 = np.exp(c)


#noisyData = [[1,2,3,4],[5,6,7,8]]
plt.plot(x,a*np.exp(b*x),"red")
ax.scatter(x, y)
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
plt.plot(x,t1*np.exp(t2*x),"green")
plt.plot(x,a*np.exp(b*x),"red")
ax1.scatter(x, y)
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
plt.plot(x,t1*np.exp(t2*x),"green")
plt.plot(x,a*np.exp(b*x),"red")
plt.plot(x,(t1-1)*np.exp(t2*x),"blue")
plt.plot(x,(t1-2)*np.exp(t2*x),"orange")
plt.plot(x,(t1+1)*np.exp(t2*x),"purple")
plt.plot(x,(t1+2)*np.exp(t2*x),"cyan")
ax2.scatter(x, y)
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
plt.plot(x,t1*np.exp(t2*x),"green")
plt.plot(x,a*np.exp(b*x),"red")
plt.plot(x,t1*np.exp((t2+1)*x),"blue")
plt.plot(x,t1*np.exp((t2+2)*x),"cyan")
plt.plot(x,t1*np.exp((t2-1)*x),"orange")
plt.plot(x,t1*np.exp((t2-2)*x),"purple")
ax3.scatter(x, y)
plt.show()
