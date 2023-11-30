import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

trsf = np.load("Efoc.npy")
ref = np.load("pol_matrix_vr.npy")
ny, nx, _ = trsf.shape
print(trsf.shape)
#I = np.sum(np.real(trsf*np.conj(trsf)), axis=-1)
I = fftshift(np.real(np.conj(trsf)*trsf))
I = np.sum(I, axis=-1)
#I = I[:, :, 0]
I /= I.max()
Iref = np.real(np.trace(ref, axis1=-1, axis2=-2))
#Iref = np.real(ref[:, :, 2, 2])
Iref /= Iref.max()
print(Iref.shape)
#TODO: Calcula segons python el camp focal!...
fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0, 0].imshow(I, cmap="hot")
ax[0, 0].set_title("Fortran")
ax[0, 1].imshow(Iref, cmap="hot")
ax[0, 1].set_title("Python")
ax[1, 0].plot(I[:, ny//2])
ax[1, 1].plot(Iref[:, ny//2])
ax[1, 1].plot(I[:, ny//2])
plt.show()
