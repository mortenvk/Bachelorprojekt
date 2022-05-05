import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
import csv

samlet_opt =genfromtxt('<var>', delimiter=',')

plt.plot(samlet_opt, label="Average optimality")
plt.axhline(y=0.0611, color='k')
plt.axhline(y=0.125, color='k')
plt.xlabel('t')
plt.ylabel('Avg. optimality')
plt.ylim(0, 0.15)
plt.legend(loc='lower right')
plt.show()
