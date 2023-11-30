from radial_coherence import RadiallyCoherentField, construct_pol_matrix, compute_pol_deg
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import sys

def process_n(N, show=False, verbose=False):
    print(f"Processing N = {N}")
    n = 256
    #xmax = 40.00   # ORIGINAL!
    xmax = 10.00
    rho = np.linspace(0, xmax, n)
    radial_fun = rho*(rho < 0.95)
    radfield = RadiallyCoherentField(radial_fun, xmax=xmax)
    # Calcula en z
    #zmax = 10      # Numero que li vaig dir a la Charo
    zmax = 4.0
    nz = 128
    zetes = np.linspace(-zmax, zmax, nz)
    P_z = np.zeros(nz)
    I_z = np.zeros(nz)
    for i, z in enumerate(zetes):
        if verbose:
            print(f"Iteration {i}; z = {z}")
        radfield.set_z(z)
        En, u = radfield.compute_field(N=N, verbose=False)
        W = construct_pol_matrix(En, En)
        P_z[i] = compute_pol_deg(W)[n//2, n//2]
        I_z[i] = np.real(np.trace(W, axis1=-1, axis2=-2))[n//2, n//2]

    np.save(f"dop_z_N-{N:02d}.npy", P_z)
    np.save(f"I_z_N-{N:02d}.npy", I_z)
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].plot(zetes, P_z)
    ax[0].set_xlabel(r"$z/\lambda$")
    ax[0].set_ylabel(r"DoP")
    ax[1].plot(zetes, I_z)
    ax[1].set_xlabel(r"$z/\lambda$")
    ax[1].set_ylabel(r"Irradiance (a.u.)")

    fig.savefig(f"dop_z_N-{N:02d}.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()

def term_calc():
    """Calcula només un terme. Les dades només entren des del terminal!"""
    if len(sys.argv) < 2:
        N = input("Nombre de termes?\n> ")
        try:
            N = int(N)
        except:
            raise ValueError("Incorrect value!")
    else:
        N = int(sys.argv[1])
        show_plots = sys.argv[2]
        show = True
        if len(sys.argv) == 3:
            show_plots = sys.argv[2]
            if show_plots == "noshow":
                show = False
    process_n(N, show=show)

def batch_process():
    """Processa diferents camps amb diferents valors de N en paral·lel 
    utilitzant multiprocessing."""
    if len(sys.argv) < 3:
        tup = input("Rang de valors de N?\n> ")
        tup = tup.split(" ")
        try:
            Nmin, Nmax = [int(n) for n in tup]
        except:
            raise ValueError("Valors incorrectes!")
    else:
        Nmin = int(sys.argv[1])
        Nmax = int(sys.argv[2])
        show = False
        if len(sys.argv) == 4:
            show_plots = sys.argv[3]
            if show_plots == "show":
                show = True
    n_cpu = mp.cpu_count()
    pool = mp.Pool(n_cpu)
    # Comença a treballar...
    pool.map(process_n, range(Nmin, Nmax+1))

if __name__ == "__main__":
    batch_process()
