import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import os


# t1 = np.array([np.arange(0, 4, 1), np.arange(0, 4, 1)])
# t2 = np.array([np.arange(0, 4, 1), np.arange(0, 4, 1)])

# print('t1.shape', t1)
# print('t2.shape', t2.T)

# print(t1)
# print(t1.dot(t2.T))

W1 = np.load("model/W1.npy")
b1 = np.load("model/b1.npy")
W2 = np.load("model/W2.npy")
b2 = np.load("model/b2.npy")

print("W1.shape", W1.shape)
print("W2.shape", W2.shape)
print("b1.shape", b1.shape)
print("b2.shape", b2.shape)

def avgW1(W1):
    out = np.zeros((784))
    print("out.shape", out)

    for row in W1:
        print('row.shape', row.shape)
        i = 0
        for v in row:
            out[i] += v
            i += 1
    out = out / 10
    
    return out.reshape((28, 28))

N = 10

X, Y = np.mgrid[0:10:complex(0, N), 0:10:complex(0, N)]
# Z = 5 * np.exp(-X**2 - Y**2)
Z = W2

# print("X.shape", X)
fig, ax = plt.subplots(2, 1)

# pcm = ax[0].pcolormesh(X, Y, Z,
#                        norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
#                                               vmin=-1.0, vmax=1.0, base=10),
#                        cmap='RdBu_r', shading='nearest')
# fig.colorbar(pcm, ax=ax[0], extend='both')

pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z),
                       shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both')


N = 28
X, Y = np.mgrid[0:28:complex(0, N), 0:28:complex(0, N)]
Z = avgW1(W1)
print("Z.shape", Z.shape)
# print('X.shape', X)

pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z),
                       shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='both')

plt.show()