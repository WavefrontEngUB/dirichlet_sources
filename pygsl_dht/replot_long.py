import numpy as np
import matplotlib.pyplot as plt

noms_dop = "dop_z_N-{:02d}.npy"
noms_I = "I_z_N-{:02d}.npy"

N = range(1, 8)
zmax = 4
first = True
for n in N:
    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    P_z = np.load(noms_dop.format(n))
    I_z = np.load(noms_I.format(n))
    if first:
        zetes = np.linspace(-zmax, zmax, P_z.shape[0])
        first = False

    ax[0].plot(zetes, P_z)
    ax[0].set_xlabel(r"$z/\lambda$")
    ax[0].set_ylabel(r"DoP")
    ax[0].set_ylim([0, 1])
    ax[0].set_title("Axial DoP")

    ax[1].plot(zetes, I_z)
    ax[1].set_xlabel(r"$z/\lambda$")
    ax[1].set_ylabel(r"Irradiance (a.u.)")
    ax[1].set_title("Axial Irradiance")

    fig.savefig(f"dop_z_N-{n:02d}.png", dpi=200, bbox_inches="tight")

    plt.close()
