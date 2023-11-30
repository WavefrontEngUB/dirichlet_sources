#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RW Partially coherent diffraction from first principles...?
"""
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.fft import fft2, ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

def convolve(f, g):
    """Convolve two 2D functions f, g."""
    fk = fft2(f, workers=-1)
    gk = fft2(g, workers=-1)
    return ifft2(gk*np.conj(fk), workers=-1)

def rw_matrix(tau, WS, Lx, Ly, f, NA, z=0, lamb=500e-6, dtype=np.complex64,
        full_matrix=False, antidiag=False):
    """
    Input:
        - tau: (Ny, Nx, 2, 2) matrix.
        - WS: idem.

    """
    # Select precision
    if dtype != np.complex64 and dtype != np.complex_:
        raise ValueError("Precision type not understood.")

    ny, nx, _, _ = tau.shape
    y, x = np.mgrid[-ny//2:nx//2, -nx//2:nx//2]
    phi = np.arctan2(y, x)
    Lxf = lamb*f*nx/4/Lx
    Lyf = lamb*f*ny/4/Ly
    thmax = np.arcsin(NA)

    # Coordinates at the GRS
    x_inf = x/x.max()*Lx
    y_inf = y/y.max()*Ly
    rho2 = x_inf*x_inf+y_inf*y_inf

    # Cosine and sine factors and E.P. mask
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    sinth2 = rho2/f/f
    sinth = np.sqrt(sinth2)
    mask = sinth < NA
    costh = np.sqrt(1-sinth2*mask)*mask
    plt.imshow(costh)
    plt.show()
    sqcos = np.sqrt(costh)

    # Build the N0 matrix
    N = np.zeros((ny, nx, 2, 3), dtype=dtype)
    N[:, :, 0, 0] = (sinphi*sinphi + cosphi*cosphi*costh)
    N[:, :, 0, 1] = sinphi*cosphi*(costh-1)
    N[:, :, 0, 2] = sinth*cosphi

    N[:, :, 1, 0] = sinphi*cosphi*(costh-1)
    N[:, :, 1, 1] = (sinphi*sinphi*costh + cosphi*cosphi)
    N[:, :, 1, 2] = sinth*sinphi

    N0 = np.matmul(tau, N)
    for j in range(2):
        for i in range(3):
            N0[:, :, j, i] = N0[:, :, j, i]/sqcos*np.exp(-2j*np.pi/lamb*z*costh)*mask

    # Create the coherency matrix and calculate its components
    W = np.zeros((ny, nx, 3, 3), dtype=dtype)
    # ONLY compute upper diagonal terms
    # Degree of coherence
    if antidiag:
        for j in range(3):
            for i in range(j, 3):
                W[:, :, j, i] = fft2(
                        np.conj(N0[:, :, 0, j])*WS[:, :, 0, 0]*N0[:, :, 0, i]+\
                        np.conj(N0[:, :, 1, j])*WS[:, :, 1, 1]*N0[:, :, 1, i]+\
                        np.conj(N0[:, :, 1, j])*WS[:, :, 1, 0]*N0[:, :, 0, i]+\
                        np.conj(N0[:, :, 0, j])*WS[:, :, 0, 1]*N0[:, :, 1, i],
                        workers=-1)
    # Intensity and polarization
    else:
        Wshift = fftshift(WS, axes=(0, 1))
        for j in range(3):
            for i in range(j, 3):
                W[:, :, j, i] = fft2(
                        convolve(N0[:, :, 0, j], N0[:, :, 0, i])*Wshift[:, :, 0, 0]+\
                        convolve(N0[:, :, 1, j], N0[:, :, 1, i])*Wshift[:, :, 1, 1]+\
                        convolve(N0[:, :, 0, j], N0[:, :, 1, i])*Wshift[:, :, 0, 1]+\
                        convolve(N0[:, :, 1, j], N0[:, :, 0, i])*Wshift[:, :, 1, 0],
                        workers=-1)
                        #2*np.real(convolve((N0[:, :, 1, j]), N0[:, :, 0, i])\
                        #    *WS[:, :, 0, 1]), workers=-1)

    # Loop end
    if full_matrix:
        W[:, :, 1, 0] = np.conj(W[:, :, 0, 1])
        W[:, :, 2, 0] = np.conj(W[:, :, 0, 2])
        W[:, :, 2, 1] = np.conj(W[:, :, 1, 2])
    return fftshift(W, axes=(0, 1))

def main_2():
    # Dades del feix i el S.O.
    ny, nx = 512, 512
    lamb = 1
    NA = 0.95
    thm = np.arcsin(NA)
    f = 5
    Lf = 8*lamb
    print("Lf = ", Lf)
    Lx = f*lamb*nx/4/Lf
    Ly = Lx/nx*ny

    y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]
    r2 = x*x+y*y
    phi = np.arctan2(y, x)
    dtype = np.complex128
    tau = np.zeros((ny, nx, 2, 2), dtype=dtype)
    # Diagonal
    r = np.sqrt(x*x+y*y)

    #tau[:, :, 0, 0] = np.exp(2j*np.pi/lamb*zernike_p(rho, phi, n, m)*A)
    #tau[:, :, 1, 1] = eval_hermite(1, x/sig)*eval_hermite(1, y/sig)*fill_factor
    #tau[:, :, 0, 0] = r*np.sin(1*phi)*np.exp(1j*phi)*fill_factor
    #tau[:, :, 0, 0] = r*np.cos(phi)*np.exp(1j*phi)*fill_factor#*np.exp(2j*np.pi/lamb*zernike_p(rho, phi, n, m)*A)
    #tau[:, :, 1, 1] = r*np.sin(phi)*np.exp(1j*phi)*fill_factor#*np.exp(2j*np.pi/lamb*zernike_p(rho, phi, n, m)*A)
    tau[:, :, 1, 1] = 1

    # Matriu de polaritzacio
    P = np.zeros((2, 2), dtype=dtype)
    #beta = np.arctan2(1, 0)
    beta = np.pi/2-0.0622
    P[0, 0] = 0
    P[1, 1] = 1
    P[0, 1] = 0
    P[1, 0] = 0
    H = np.zeros((ny, nx, 2, 2), dtype=dtype)
    print(np.linalg.det(P))
    """
    H[ny//2, nx//2, 0, 0] = 1
    H[ny//2, nx//2, 1, 1] = 1
    """
    H[:, :, 0, 0] = 1
    H[:, :, 1, 1] = 1
    WS = P@H

    W = rw_matrix(tau, WS, Lx, Ly, f, NA, lamb=lamb, full_matrix=True, dtype=dtype,
            antidiag=False)
    W[:] = W/(np.real(W[:, :, 0, 0]+W[:, :, 1, 1]+W[:, :, 2, 2]).max())

    # Determine the degree of polarization
    Itot = np.real(np.trace(W, axis1=-2, axis2=-1)) # Tr(W) dona la intensitat
    vmax = Itot.max()

    # Metode nou, P^2 = 3/2 * (tr W^2/(tr W)^2 - 1/3)
    W2 = W@W    # OMG O_O
    tr2 = np.trace(W2, axis1=-2, axis2=-1)
    P = np.sqrt(3/2*(tr2/(Itot*Itot)-1/3))

    """
    s3 = np.imag(W[:, :, 1, 0]-W[:, :, 0, 1])
    s2 = np.real(W[:, :, 1, 0]+W[:, :, 0, 1])
    s1 = np.real(W[:, :, 0, 0]-W[:, :, 1, 1])
    plt.imsave("0.png", np.real(W[:, :, 0, 0]), vmin=0, vmax=vmax, cmap="gray")
    plt.imsave("90.png", np.real(W[:, :, 1, 1]), vmin=0, vmax=vmax, cmap="gray")
    plt.imsave("45.png", np.real(Itot+s2), vmin=0, vmax=2*vmax, cmap="gray")
    plt.imsave("135.png", np.real(Itot-s2), vmin=0, vmax=2*vmax, cmap="gray")
    plt.imsave("levo.png", np.real(Itot+s3), vmin=0, vmax=2*vmax, cmap="gray")
    plt.imsave("dextro.png", np.real(Itot-s3), vmin=0, vmax=2*vmax, cmap="gray")
    """
    Lf *= 1e3
    cmap = "gray"
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(np.real(W[:, :, 0, 0]), cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmax=vmax)
    ax[0,0].set_title(r"$W_{00}$")
    ax[0,1].imshow(np.real(W[:, :, 1, 1]), cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmax=vmax)
    ax[0,1].set_title(r"$W_{11}$")
    ax[1,0].imshow(np.real(W[:, :, 2, 2]), cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmax=None)
    ax[1,0].set_title(r"$W_{22}$")
    ax[1,1].imshow(Itot, cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmax=vmax)
    ax[1,1].set_title(r"$\mathrm{Tr}(W)$")
    plt.tight_layout()
    print(np.real(W[:, :, 2, 2]).max()/np.real(W[:, :, 1, 1]+W[:, :, 0, 0]+W[:, :, 2, 2]).max())
    #fig.savefig("Focal_reg.png", dpi=200, bbox_inches="tight")

    """
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_title("Polarization")
    ax2.imshow(np.real(P), vmin=0, vmax=1, extent=[-Lf, Lf, -Lf, Lf])
    """

    fig3, ax3 = plt.subplots(1, 1)
    ax3.set_title("Intensity")
    ax3.imshow(Itot, vmin=0, extent=[-Lf, Lf, -Lf, Lf], cmap="hot")
    plt.show()

    #np.save("pol_matrix_vr.npy", W)

if __name__ == "__main__":
    main_2()
