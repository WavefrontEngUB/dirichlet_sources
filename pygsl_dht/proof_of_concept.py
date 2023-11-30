import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
"""
Recreate experimentally the dirichlet degree of coherence.
"""
# TODO: Make it work

def assign_phases(E, phases, r2_max=2):
    n, = phases.shape
    dphi = 2 * np.pi / n

    ny, nx = E.shape
    for j in range(ny):
        for i in range(nx):
            y = ny // 2 - j
            x = i - nx // 2
            r2 = x * x + y * y
            if r2 > r2_max:
                E[j, i] = 0
                continue

            phi = np.arctan2(y, x)
            idx_phase = int(phi // dphi)
            E[j, i] = np.exp(1j * phases[idx_phase]) * np.sqrt(r2)

def main():
    n = 256
    num_iter = 1000
    r = 16
    r2 = r * r

    irradiance = np.zeros((n, n))
    E = np.zeros((n, n), dtype=np.complex_)
    y, x = np.mgrid[-n//2:n//2, -n//2:n//2]
    phi = np.arctan2(y, x)
    arg_E = np.exp(1j * phi)
    for k in range(num_iter):
        phases = 2 * np.pi * np.random.rand(n)
        assign_phases(E, phases, r2_max=r2)
        #E[:] *= arg_E
        E_focal = fftshift(fft2(ifftshift(E)))
        irradiance += np.real(np.conj(E_focal) * E_focal)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(irradiance)
    plt.show()

if __name__ == "__main__":
    main()
