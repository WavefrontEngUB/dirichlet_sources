import numpy as np
import matplotlib.pyplot as plt
import sys

k = int(sys.argv[1])
rad = np.load(f"{k}_radial-irr.npy")
pol = np.load(f"{k}_radial-dop.npy")
ny, nx, nz = rad.shape

fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0, 0].imshow(np.fft.fftshift((rad[:, :, nz//2])))
ax[0, 0].set_title("z = 0")
ax[0, 1].imshow(np.fft.fftshift((rad[:, :, 0])))
ax[0, 1].set_title("z = z_min")

ax[1, 0].imshow(np.fft.fftshift((pol[:, :, nz//2])), vmin=0, vmax=1, cmap="bwr")
ax[1, 1].imshow(np.fft.fftshift((pol[:, :, 0])), vmin=0, vmax=1, cmap="bwr")

plt.show()
