import pydht
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

class RadiallyCoherentField():
    def __init__(self, radial_fun, xmax=1.):
        """a=mida de la pupil·la"""
        self.n = len(radial_fun)
        self.xmax = xmax
        self.radial_fun = radial_fun
        self.Ap = np.zeros(self.n, dtype=np.complex128)
        self.An = np.zeros(self.n, dtype=np.complex128)
        self.A = np.zeros(self.n, dtype=np.complex128)

        # Transformacions
        self.dht1 = pydht.DHT(self.n)
        self.dht2 = pydht.DHT(self.n)
        self.dht3 = pydht.DHT(self.n)

        # Prepara els kernels
        self.rho = None
        self.sqrt = None
        self.set_z()
        self._prepare_vectors()

    def set_z(self, z=0, lamb=1):
        if not isinstance(self.rho, np.ndarray):
            self.rho = np.linspace(0, self.xmax, self.n)
        mask = self.radial_fun>0
        if not isinstance(self.sqrt, np.ndarray):
            self.sqrt = np.sqrt(1-self.rho*self.rho*mask)*mask
        prop_term = np.exp(-2j*np.pi/lamb*z*self.sqrt)
        pr_r = np.real(prop_term)
        pr_i = np.imag(prop_term)
        denom = np.sqrt(self.sqrt)

        # TODO: Separa en parts real i imaginària en cas de propagar en z...
        self.k1_r = self.sqrt*self.radial_fun*pr_r
        self.k1_i = self.sqrt*self.radial_fun*pr_i
        self.k1_r[mask] /= denom[mask]
        self.k1_i[mask] /= denom[mask]
        self.k2_r = self.rho*self.radial_fun*pr_r
        self.k2_i = self.rho*self.radial_fun*pr_i
        self.k2_r[mask] /= denom[mask]
        self.k2_i[mask] /= denom[mask]

    def _prepare_vectors(self):
        self.u1 = np.array((1j, 1, 0))
        self.u2 = np.array((-1j, 1, 0))
        self.u3 = np.array((0, 0, 1))

    def compute_field(self, N=1, verbose=False):
        """Calculem el camp amb un nombre de termes N."""
        E_tot = np.zeros((self.n, self.n, 3, 2*N+1), dtype=np.complex128)
        y, x = np.mgrid[-self.n//2:self.n//2, -self.n//2:self.n//2]
        r = np.sqrt(y*y+x*x)
        r_i = np.int_(r)
        phi = np.arctan2(y, x)
        exphi = np.exp(1j*phi)
        nexphi = np.exp(-1j*phi)
        
        # Expansions radials
        Axr = np.zeros((self.n, self.n), dtype=np.complex128)
        Axrp = np.zeros((self.n, self.n), dtype=np.complex128)
        Axrn = np.zeros((self.n, self.n), dtype=np.complex128)
        # TODO: Optimitza l'ús de la cpu. No ens cal calcular mil cops la mateixa transformada...
        # Calcula totes les transformades que utilitzarem per a construir E
        A_n = np.zeros((self.n, N+3), dtype=np.complex128)
        C_n = np.zeros((self.n, N+1), dtype=np.complex128)
        for n in range(0, N+2):
            self.dht1.init(float(n), self.xmax)
            self.An[:]  = self.dht1.apply(self.k1_r)
            self.An[:] += 1j*self.dht1.apply(self.k1_i)
            A_n[:, n] = self.An
            if n <= N:
                self.dht2.init(float(n), self.xmax)
                self.A[:]  = self.dht2.apply(self.k2_r)
                self.A[:] += 1j*self.dht2.apply(self.k2_i)
                C_n[:, n] = self.A
        # Ara fem el càlcul de En, el n-èssim mode del camp parcialment coherent
        base1 = np.zeros_like(self.A)
        base2 = np.zeros_like(self.A)
        for n in range(-N, N+1):
            # An-1 i An+1 respectivament
            index1 = n-1
            index2 = n+1
            # Canviem el signe de la funció segons convingui
            if index1 < 0:
                base1 = np.negative(A_n[:, abs(index1)])
            else:
                base1 = A_n[:, abs(index1)]

            if index2 < 0:
                base2 = np.negative(A_n[:, abs(index2)])
            else:
                base2 = A_n[:, abs(index2)]
            # C_n, res fancy
            if n < 0:
                base3 = np.negative(C_n[:, abs(n)])
            else:
                base3 = C_n[:, abs(n)]

            # Construïm els camps 2D a partir de les transformades 1D
            Axr[:] = base3[r_i]
            Axrn[:] = base1[r_i]
            Axrp[:] = base2[r_i]

            E_tot[:, :, :, N+n] = \
                    np.outer(np.exp(1j*n*phi), np.array((1,1,1)))\
                        .reshape((self.n, self.n, 3))*\
                    (1j)**n*(
                    np.outer(nexphi*Axrn, self.u1).reshape((self.n, self.n, 3)) +\
                    np.outer(exphi*Axrp, self.u2).reshape((self.n, self.n, 3))+\
                    2*np.outer(Axr, self.u3).reshape((self.n, self.n, 3))
                    )
        """
        for i in range(-N, N+1):
            # TODO
            # Configura les transformacions que farem
            if verbose:
                print(f"Computing n = {i}th term")
            n = i
            n_p = i-1
            n_n = i+1

            self.dht1.init(float(abs(n_n)), self.xmax)
            self.dht2.init(float(abs(n)), self.xmax)
            self.dht3.init(float(abs(n_p)), self.xmax)

            self.A[:] = self.dht2.apply(self.k2_r)
            self.A[:] += 1j*self.dht2.apply(self.k2_i)
            if n < 0:
                self.A *= (-1)**abs(n)
            self.An[:] = self.dht1.apply(self.k1_r)
            self.An[:] += 1j*self.dht1.apply(self.k1_i)
            if n_n < 0:
                self.An *= (-1)**abs(n_n)
            self.Ap[:] = self.dht3.apply(self.k1_r)
            self.Ap[:] += 1j*self.dht3.apply(self.k1_i)
            if n_p < 0:
                self.Ap *= (-1)**abs(n_p)

            Axr[:] = self.A[r_i]
            Axrn[:] = self.An[r_i]
            Axrp[:] = self.Ap[r_i]

            # TODO: Comprova que els productes pels vectors unitaris estiguin bé
            E_tot[:, :, :, N+i] = \
                    np.outer(np.exp(1j*i*phi), np.array((1,1,1)))\
                        .reshape((self.n, self.n, 3))*\
                    (1j)**n*(
                    np.outer(exphi*Axrn, self.u1).reshape((self.n, self.n, 3)) +\
                    np.outer(nexphi*Axrp, self.u2).reshape((self.n, self.n, 3))+\
                    2*np.outer(Axr, self.u3).reshape((self.n, self.n, 3))
                    )
        """
        # Get focal field coordinates
        u = self.dht1.get_x_samples()
        return E_tot, u

def construct_pol_matrix(a, b):
    if a.shape != b.shape:
        raise ValueError("Array shapes must be equal!")
    ny, nx, n, nt = a.shape
    
    # Construeix nous vectors perquè puguem multiplicar els arrais
    ap = np.zeros((ny, nx, nt, n, 1), dtype=a.dtype)
    bp = np.zeros((ny, nx, nt, 1, n), dtype=b.dtype)

    # TODO: Mira si es pot fer d'una forma... més elegant.
    for i in range(nt):
        ap[:, :, i, 0, 0] = np.conj(a[:, :, 0, i])
        ap[:, :, i, 1, 0] = np.conj(a[:, :, 1, i])
        ap[:, :, i, 2, 0] = np.conj(a[:, :, 2, i])
        bp[:, :, i, 0, 0] = b[:, :, 0, i]
        bp[:, :, i, 0, 1] = b[:, :, 1, i]
        bp[:, :, i, 0, 2] = b[:, :, 2, i]

    return np.sum(np.matmul(ap, bp), axis=-3)

def compute_pol_deg(W):
    # Computa el grau de polarització 3D segons T. Setälä
    num = np.trace(np.real(W*W), axis1=-1, axis2=-2)
    denom = np.trace(np.real(W), axis1=-1, axis2=-2)
    denom *= denom
    return np.sqrt(1.5*(num/denom-1/3))

def main():
    import sys
    n = 512
    xmax = 10.00
    rho = np.linspace(0, xmax, n)
    #radial_fun = rho*(rho < 0.95)
    radial_fun = rho*(rho < 0.95).astype(np.float64)
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    radfield = RadiallyCoherentField(radial_fun, xmax=xmax)
    En, u = radfield.compute_field(N=N, verbose=True)
    rmax = u.max()/(2*np.pi)  # Mida de la regió focal en longituds d'ona
    print(f"{rmax:.3g} lambda")

    # Matriu de polarització
    W = construct_pol_matrix(En, En)

    #I = np.sum(np.real(np.conj(En)*En), axis=(-1, -2))
    I = np.real(np.trace(W, axis1=-1, axis2=-2))

    # DoP
    dop2 = compute_pol_deg(W)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].imshow(I, cmap="hot", extent=[-rmax, rmax, -rmax, rmax])
    ax[0].set_title(f"{N}-term irradiance")
    ax[0].set_xlabel("$x/\lambda$")
    ax[0].set_ylabel("$y/\lambda$")
    pcm = ax[1].imshow(dop2, cmap="bwr", extent=[-rmax, rmax, -rmax, rmax], vmin=0, vmax=1)
    ax[1].set_title("3D DoP")
    ax[1].set_xlabel("$x/\lambda$")
    ax[1].set_ylabel("$y/\lambda$")

    # Colorbar
    fig.colorbar(pcm, ax=ax[1], shrink=0.49)

    fig.savefig(f"{N}-term_result.png", dpi=200, bbox_inches="tight")
    #plt.show()

if __name__ == "__main__":
    main()
