import numpy as np
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "dejavuserif"
import matplotlib.pyplot as plt

def main():
    n = 512
    phi = np.linspace(0, 2 * np.pi, n)
    N_modes = 1, 5, 20

    fig, ax = plt.subplots(2, 2, constrained_layout=True, 
            figsize=(6, 6), subplot_kw={"projection":"polar"})
    ax = ax.flatten()

    rhos = np.ones_like(phi)
    phi_rad = np.outer(phi, rhos)

    # Show the profiles in a separate graph...
    fig.delaxes(ax[-1])
    cartax = fig.add_axes([0.555, 0.086, 0.35, 0.34])

    for i, N in enumerate(N_modes):
        # Computing the cohdeg, which is independent of radial coord
        mu2 = np.sin((N + .5) * phi) 
        np.divide(mu2, np.sin(phi * .5), out=mu2)
        np.multiply(mu2, mu2, out=mu2)
        np.divide(mu2, (2 * N + 1)**2, out=mu2)

        mu2_rad = np.outer(mu2, rhos)
        mu2_rad[np.isnan(mu2_rad)] = 1.0

        pcm = ax[i].pcolormesh(phi, np.linspace(0, 1, n), mu2_rad.T, cmap="bwr", vmin=0, vmax=1)
        ax[i].set_title(f"N = {N}")
        ax[i].yaxis.set_ticks(np.linspace(0, 1, 3))
        
        mu2[mu2 > 1.] = 1.
        cartax.plot(phi / np.pi * 180, mu2, label=f"N = {N}")
    
    cartax.set_xlabel(r"$\phi_2-\phi_1(^\circ)$")
    cartax.set_ylabel(r"$\mu^2_{2D}(\phi_2-\phi_1)$")
    cartax.legend(loc="best")
    fig.colorbar(pcm, ax=ax[1], shrink=.9)

    fig.savefig("mu2dt.png", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
