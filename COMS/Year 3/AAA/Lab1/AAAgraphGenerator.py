import matplotlib.pyplot as plt
import sys

file = sys.argv[1]
title = sys.argv[2]

plt.plotfile(file, ('input', 'time'))
plt.xlabel('input size')
plt.ylabel('time (ms)')
plt.title(title);
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.savefig(title +'.pdf')
