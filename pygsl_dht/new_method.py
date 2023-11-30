import numpy as np
import matplotlib.pyplot as plt
import pydht

class DHTComputer:
    def __init__(self, npoints, z, n_z, max_mode, xmax):
        self.npoints = npoints
        self.zetes = np.linspace(-z, z, n_z)
        self.n_z = n_z
        self.max_mode = max_mode
        self.xmax = float(xmax)

        # Objecte de transformació
        self.dht = pydht.DHT(npoints)

class RadialComputer(DHTComputer):
    def __init__(self, npoints, z, n_z, max_mode, xmax):
        super().__init__(npoints, z, n_z, max_mode, xmax)
        self.A_n = np.zeros((npoints, max_mode+3, n_z), dtype=np.complex_)
        self.C_n = np.zeros((npoints, max_mode+1, n_z), dtype=np.complex_)

    def compute_aux(self, radial_function):
        H = np.zeros(self.npoints, dtype=np.complex_)
        sqrt = np.zeros(self.npoints, dtype=np.float_)
        k1_r = np.zeros_like(sqrt)
        k1_i = np.zeros_like(sqrt)
        k2_r = np.zeros_like(sqrt)
        k2_i = np.zeros_like(sqrt)
        for i in range(len(self.zetes)):
            zi = self.zetes[i]
            for j in range(0, self.max_mode+3):
                # Prepara la transformada per al mode donat
                self.dht.init(abs(float(j)), self.xmax)
                # Aconsegueix les coordenades a la pupil·la d'entrada
                x = self.dht.get_x_samples()
                mask = x < 1.0
                sqrt[mask] = np.sqrt(1-x[mask]**2)
                # Creem cada Kernel...
                H[:] = np.exp(-2j*np.pi*sqrt*zi)
                k1_r[:] = np.sqrt(sqrt)*x
                k1_i[:] = k1_r*np.imag(H)
                k1_r[:] = k1_r*np.real(H)

                k2_r[mask] = x[mask]/np.sqrt(sqrt[mask])*radial_function[mask]*x[mask]
                k2_i[:] = k2_r*np.imag(H)
                k2_r[:] = k2_r*np.real(H)

                # Comencem les transformades...
                self.A_n[:, j, i] = self.dht.apply(k1_r)
                self.A_n[:, j, i] += 1j*self.dht.apply(k1_i)

                if j <= self.max_mode:
                    self.C_n[:, j, i] = self.dht.apply(k2_r)
                    self.C_n[:, j, i] += 1j*self.dht.apply(k2_i)
                # Netejo els kernels...
                k1_r[:] = 0
                k1_i[:] = 0
                k2_r[:] = 0
                k2_i[:] = 0
                sqrt[:] = 0

    def compute_axial_irradiance(self):
        irradiances = np.zeros((self.npoints, self.max_mode+1, self.n_z))
        for i in range(-self.max_mode, self.max_mode+1):
            idx_pos = i+1
            idx_neg = i-1
            s_pos = 1 if idx_pos >= 1 else -1
            s_neg = 1 if idx_neg >= 1 else -1
            Ex = 1j**i*(self.A_n[:, abs(idx_pos), :]*1j*s_pos - \
                     1j*self.A_n[:, abs(idx_neg), :]*s_neg)
            Ey = 1j**i*(self.A_n[:, abs(idx_pos), :]*s_pos - \
                        self.A_n[:, abs(idx_neg), :]*s_neg)
            Ez = 1j**i*2*self.C_n[:, abs(i), :]

            irradiances[:, abs(i), :] = np.conj(Ex)*Ex+np.conj(Ey)*Ey+np.conj(Ez)*Ez

        return irradiances

    def compute_axial_dop(self):
        a2 = np.real(np.conj(self.A_n[0, self.max_mode-1, :])*self.A_n[0, self.max_mode-1, :])
        c2 = np.real(np.conj(self.A_n[0, self.max_mode-1, :])*self.A_n[0, self.max_mode-1, :])
        num = abs(a2-2*c2)
        denom = a2+c2
        return .5*num/denom

    def save_results(self, fname):
        np.savez(fname, A_n=self.A_n, C_n=self.C_n)

def main():
    npoints = 256
    n_z = 129
    xmax = 10
    max_mode = 1
    z = 4
    x = np.linspace(0, xmax, npoints)
    radcomputer = RadialComputer(npoints, z, n_z, max_mode, xmax)

    radcomputer.compute_aux(x*(x<0.95))
    radcomputer.save_results("an_i_cn.npz")

    irradiances = radcomputer.compute_axial_irradiance()
    np.save("axial_irradiance.npy", irradiances)

    dop = radcomputer.compute_axial_dop()

    np.savez("axial_radial_dht.npz", dop=dop, irr=irradiances)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].plot(irradiances[0, 0, :])
    ax[1].plot(dop)
    plt.show()

if __name__ == "__main__":
    main()
