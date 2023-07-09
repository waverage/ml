import numpy as np
from matplotlib import pyplot as plt

def ReLU(Z):
    return np.maximum(Z, 0)

def deriv_ReLU(Z):
    return Z > 0

# Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)

# fig, ax = plt.subplots()
# ax.plot(t, s)

# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()

# plt.show()

fig, ax = plt.subplots()

t2 = np.arange(-1, 1.0, 0.01)
s2 = ReLU(t2)
ax.plot(t2, s2, label="ReLU")

s3 = deriv_ReLU(t2)
ax.plot(t2, s3, label="ReLU deriv")

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

plt.show()