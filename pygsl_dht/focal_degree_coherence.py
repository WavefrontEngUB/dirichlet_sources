import numpy as np
from pydht import DHT
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"
import matplotlib.pyplot as plt


def kernG(rho, NA):
    mask = rho <= NA
    k = rho * mask
    arg = 1 - rho * rho * mask
    np.divide(k, arg ** .25, where=mask, out=k)
    return k


def kernA(rho, NA):
    mask = rho <= NA
    arg = 1 - rho * rho * mask
    kern = rho * np.sqrt(arg) * mask
    np.divide(kern, arg ** .25, where=mask, out=kern)
    return kern


def kernC(rho, NA):
    mask = rho <= NA
    arg = 1 - rho * rho * mask
    kern = rho * rho * mask
    np.divide(kern, arg ** .25, where=mask, out=kern)
    return kern


def f(rho):
    return rho 


def compute_cohdeg(N, NA, rho_max, n_lat, kind="azimuthal", rho_lim=-1, vmax=1):
    # Initializing dht, needed to obtain the sampling point in direct and reciprocal space
    dht = DHT(n_lat)

    if rho_lim == -1:
        rho_lim = rho_max
    if kind == "azimuthal":
        modes = np.zeros((n_lat, N + 2), dtype=np.float64)
    elif kind == "radial":
        modesA = np.zeros((n_lat, N + 2), dtype=np.float64)
        modesC = np.zeros((n_lat, N + 2), dtype=np.float64)
    sample_points = np.zeros(n_lat, dtype=np.float64)
    for n in range(0, N + 2):
        dht.init(n, rho_max)

        rho = dht.get_x_samples()
        if kind == "azimuthal":
            kernel = kernG(rho, NA)
        
            g_n = dht.apply(kernel * f(rho))
            # Saving the results for each loop iteration
            k_samples = dht.get_k_samples()
            if n == 0:
                sample_points[:] = np.linspace(0, k_samples.max(), len(k_samples))
            modes[:, n] = np.interp(sample_points, k_samples, g_n)

        elif kind == "radial":
            k1 = kernA(rho, NA)
            a_n = dht.apply(k1 * f(rho))
            k2 = kernC(rho, NA)
            c_n = dht.apply(k2 * f(rho))
            k_samples = dht.get_k_samples()
            if n == 0:
                sample_points[:] = np.linspace(0, k_samples.max(), len(k_samples))
            modesA[:, n] = np.interp(sample_points, k_samples, a_n)
            modesC[:, n] = np.interp(sample_points, k_samples, c_n)


    if kind == "azimuthal":
        mu2, radial_irr, phi = azimuthal_case(n_lat, modes, sample_points, N)
    elif kind == "radial":
        mu2, radial_irr, phi = radial_case(n_lat, modesA, modesC, sample_points, N)

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, constrained_layout=True, figsize=(8, 4))
    pcm = ax[0].pcolormesh(phi, sample_points / (2 * np.pi), mu2.T, cmap="bwr", vmin=0, vmax=vmax)
    fig.colorbar(pcm, ax=ax[0], shrink=0.9)

    ax[0].set_title(f"$\mu^2 (r, \phi_2 - \phi_1, z = 0)$")
    ax[1].set_title(r"$I(r, \phi, z = 0)$")
    #ax[1].pcolormesh(phi, sample_points / (2 * np.pi), radial_irr.T, cmap="magma")

    # DELETE AXIS TO PLOT PROFILE OF MU2
    fig.delaxes(ax[1])
    cartax = fig.add_axes([0.705, 0.156, 0.29, 0.7])
    r_norm = sample_points / (2 * np.pi)
    idx = np.argwhere(np.isclose(r_norm, 0.45, rtol=0.01))[0][0]
    id2 = np.argwhere(np.isclose(r_norm, 1.00, rtol=0.01))[0][0]
    r0 = sample_points[idx] / (2 * np.pi)
    r1 = sample_points[id2] / (2 * np.pi)
    cartax.plot(phi / np.pi * 180, mu2[:, idx], label=f"r = {r0:.2g}" + r"$\lambda$" )
    cartax.plot(phi / np.pi * 180, mu2[:, id2], label=f"r = " + r"$\lambda$" )
    cartax.set_ylim([0, 0.2])
    cartax.yaxis.set_ticks(np.linspace(0, 1, 6))
    cartax.set_xlabel(r"$\phi_2 - \phi_1 (^\circ)$")
    cartax.set_ylabel(r"$\mu^2(r, \phi_2 - \phi_1, 0)$")
    cartax.legend(loc="best")
    #rlabels = ax[1].get_ymajorticklabels()
    #for label in rlabels:
    #    label.set_color("white")
    rlabels = ax[0].get_ymajorticklabels()
    for label in rlabels:
        label.set_color("black")

    #for eix in ax:
    #    eix.set_ylim(0, rho_lim)
    ax[0].set_ylim(0, rho_lim)
    ax[0].yaxis.set_ticks(np.linspace(0, rho_lim, 5))

    # FIXME: Test to see if the 3d poldeg coincides with the mu^2...
    mu_phiphi = mu2[0, :]
    P3d2 = 1.5 * mu_phiphi - .5
    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(np.linspace(0, rho_max, len(mu_phiphi)), np.sqrt(P3d2))
    ax2.set_xlim(0, rho_lim)

    #fig.savefig(f"coherence-azimuthal-N{N}.pdf", dpi=100, bbox_inches="tight")
    fig.savefig(f"coherence-{kind}-N{N}.png", dpi=200, bbox_inches="tight")

    plt.show()

def radial_case(n_lat, modesA, modesC, sample_points, N):
    irradiance = np.zeros(n_lat, dtype=np.float64)
    mu2 = np.zeros((n_lat, n_lat), dtype=np.float64)
    phi = np.linspace(0, 2 * np.pi, n_lat)
    A = np.zeros((n_lat, n_lat), dtype=np.complex128)
    B = np.zeros((n_lat, n_lat), dtype=np.complex128)
    C = np.zeros((n_lat, n_lat), dtype=np.complex128)
    R = np.zeros((n_lat, n_lat), dtype=np.complex128)
    T = np.zeros((n_lat, n_lat), dtype=np.complex128)
    S = np.zeros((n_lat, n_lat), dtype=np.complex128)
    c_ph1 = np.zeros(n_lat, dtype=np.complex128)
    u1 = np.zeros(n_lat, dtype=np.complex128)
    u2 = np.zeros(n_lat, dtype=np.complex128)
    u3 = np.zeros(n_lat, dtype=np.complex128)
    u4 = np.zeros(n_lat, dtype=np.complex128)
    for n in range(0, N + 1):
        c_ph1 = np.exp(-1j * n * phi)
        if n == 0:
            irradiance += modesA[:, abs(n+1)] ** 2 + modesC[:, n] ** 2 
            A[:, :] += np.outer(np.ones_like(u1), modesA[:, n+1] ** 2)
            B[:, :] += np.outer(np.ones_like(u1), modesA[:, n-1] ** 2)
            C[:, :] += np.outer(np.ones_like(u1), modesC[:, n] ** 2)
            R[:, :] += np.outer(np.ones_like(u1), -modesA[:, abs(n-1)] * modesA[:, n+1])
            T[:, :] += np.outer(np.ones_like(u1),  modesA[:, abs(n+1)] * modesC[:, n])
            S[:, :] += np.outer(np.ones_like(u1), -modesA[:, abs(n-1)] * modesC[:, n])
            continue
        if n < N + 1:
            irradiance += modesA[:, abs(n+1)] ** 2 + 2 * modesC[:, n] **2 + modesA[:, abs(-n+1)] ** 2

        prod = np.outer(c_ph1 , modesC[:, n] * modesC[:, n])
        C[:, :] += prod + np.conj(prod)

        n1 = n + 1
        n2 = -n + 1
        u1 = modesA[:, n1] * np.sign(n1)
        u2 = modesA[:, abs(n2)] * np.sign(n2)
        A[:, :] += np.outer(        c_ph1,  modesA[:, n1] ** 2) + \
                   np.outer(np.conj(c_ph1), modesA[:, abs(n2)] ** 2)
        T[:, :] += np.outer(        c_ph1,   modesC[:, n] * u1) + \
                   np.outer(np.conj(c_ph1), -modesC[:, n] * u2)

        n1 = n - 1
        n2 = -n - 1
        u3 = modesA[:, abs(n1)] * np.sign(n1)
        u4 = modesA[:, abs(n2)] * np.sign(n2)
        B[:, :] += np.outer(        c_ph1,  modesA[:, n1] ** 2) + \
                   np.outer(np.conj(c_ph1), modesA[:, abs(n2)] ** 2)
        S[:, :] += np.outer(        c_ph1,   modesC[:, n] * u3) + \
                   np.outer(np.conj(c_ph1), -modesC[:, n] * u4)

        R[:, :] += np.outer(        c_ph1,  u1 * u3) + \
                   np.outer(np.conj(c_ph1), u2 * u4)
    mu2[:] =     np.real(np.conj(A) * A) +\
                 np.real(np.conj(B) * B) +\
             4 * np.real(np.conj(C) * C) +\
             2 * np.real(np.conj(R) * R) +\
             4 * np.real(np.conj(S) * S) +\
             4 * np.real(np.conj(T) * T)

    radial_irr = 2 * np.outer(np.ones_like(irradiance), irradiance)
    np.divide(mu2, radial_irr ** 2, out=mu2)

    return mu2, radial_irr, phi


def azimuthal_case(n_lat, modes, sample_points, N):
    irradiance = np.zeros(n_lat, dtype=np.float64)
    mu2 = np.zeros((n_lat, n_lat), dtype=np.float64)
    phi = np.linspace(0, 2 * np.pi, n_lat)
    # Computing now the important quantities, degree of coherence, polarization and irr.
    G = np.zeros((n_lat, n_lat), dtype=np.complex128)
    F = np.zeros((n_lat, n_lat), dtype=np.complex128)
    U = np.zeros((n_lat, n_lat), dtype=np.complex128)
    c_ph1 = np.zeros(n_lat, dtype=np.complex128)
    u1 = np.zeros(n_lat, dtype=np.complex128)
    u2 = np.zeros(n_lat, dtype=np.complex128)
    u3 = np.zeros(n_lat, dtype=np.complex128)
    u4 = np.zeros(n_lat, dtype=np.complex128)
    for n in range(0, N + 1):
        if n == 0:
            irradiance += modes[:, abs(n+1)] ** 2
            sign = np.sign(n-1)
            G[:, :] += np.outer(np.ones_like(u1), modes[:, n+1] ** 2)
            F[:, :] += np.outer(np.ones_like(u1), modes[:, abs(n-1)] ** 2)
            U[:, :] += np.outer(np.ones_like(u1), modes[:, abs(n-1)] * modes[:, n+1] * sign)
            continue

        if n < N + 1:
            irradiance += modes[:, abs(n+1)] ** 2 + modes[:, abs(-n+1)] ** 2

        n1 = n + 1
        n2 = -n + 1
        c_ph1[:] = np.exp(-1j * n * phi)
        u1 = modes[:, n1] * np.sign(n1)
        u2 = modes[:, abs(n2)] * np.sign(n2)
        G[:, :] += np.outer(        c_ph1,  modes[:, n1] ** 2) + \
                   np.outer(np.conj(c_ph1), modes[:, abs(n2)] ** 2)

        n1 = n - 1
        n2 = -n - 1
        u3 = modes[:, abs(n1)] * np.sign(n1)
        u4 = modes[:, abs(n2)] * np.sign(n2)
        F[:, :] += np.outer(        c_ph1,  modes[:, abs(n1)] ** 2) + \
                   np.outer(np.conj(c_ph1), modes[:, abs(n2)] ** 2)

        U[:, :] += np.outer(        c_ph1,  u1 * u3) + \
                   np.outer(np.conj(c_ph1), u2 * u4)

    irradiance *= 2
    mu2[:] = np.real(np.conj(G) * G) + \
             np.real(np.conj(F) * F) + \
             2 * np.real(np.conj(U) * U)
    
    radial_irr = np.outer(np.ones_like(irradiance), irradiance)
    np.divide(mu2, 1 * radial_irr ** 2, out=mu2)

    return mu2, radial_irr, phi

def main():
    N = 1
    NA = 0.95
    rho_max = 40
    rho_lim = 2
    kind = "radial"
    n_lat = 2048

    compute_cohdeg(N, NA, rho_max, n_lat, kind=kind, vmax=1, rho_lim=rho_lim)

if __name__ == "__main__":
    main()
