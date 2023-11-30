import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from scipy.integrate import romb, quad

def compute_n_transform(fun, rho):
    pass

def compute_radial(NA, zetes, n=257):
    #rho = np.linspace(0, NA, n)
    n_z = len(zetes)
    res = np.zeros((n_z, 3), dtype=np.complex_)
    for i in range(n_z):
        """
        kern = np.exp(-1j*2*np.pi*zetes[i]*np.sqrt(1-rho*rho))*rho*rho/np.sqrt(1-rho*rho)
        res[i, 0] = romb((kern)*np.sqrt(1-rho*rho))

        res[i, 1] = 2*romb((kern)*rho)

        res[i, 2] = romb((kern))
        """
        res[i, 0] =     quad(kern1r, 0, NA, args=(0, zetes[i]))[0]
        res[i, 0] += 1j*quad(kern1i, 0, NA, args=(0, zetes[i]))[0]

        res[i, 1] =   2*quad(kern2r, 0, NA, args=(0, zetes[i]))[0]
        res[i, 1] += 2j*quad(kern2i, 0, NA, args=(0, zetes[i]))[0]

        res[i, 2] =     quad(kern0r, 0, NA, args=(0, zetes[i]))[0]
        res[i, 2] += 1j*quad(kern0i, 0, NA, args=(0, zetes[i]))[0]
    return res

# TODO: Utilitza un millor mètode d'integració
def kern0r(rho, r, z):
    kern = np.cos(-2*np.pi*z*np.sqrt(1-rho*rho))*rho*rho/np.sqrt(np.sqrt(1-rho*rho))
    return kern

def kern0i(rho, r, z):
    kern = np.sin(-2*np.pi*z*np.sqrt(1-rho*rho))*rho*rho/np.sqrt(np.sqrt(1-rho*rho))
    return kern

def kern1r(rho, r, z):
    return kern0r(rho, r, z)*np.sqrt(1-rho*rho)

def kern1i(rho, r, z):
    return kern0i(rho, r, z)*np.sqrt(1-rho*rho)

def kern2r(rho, r, z):
    return kern0r(rho, r, z)*rho

def kern2i(rho, r, z):
    return kern0i(rho, r, z)*rho

def compute_dop(res):
    a2 = np.real(np.conj(res[:, 0])*res[:, 0])
    #a2 -= a2.min()
    a = np.sqrt(a2)
    c2 = np.real(np.conj(res[:, 1])*res[:, 1])*.25
    #c2 -= c2.min()
    c = np.sqrt(c2)
    #dop = np.sqrt(c2*(c2-a2))/abs(a2+c2)
    # NEW: 21/09/2022
    dop = .5*abs((a2-2*c2)/(a2+1*c2))
    #dop = .5*abs((a*a-2*c*c)/(c*c+a*a))
    return dop

def main():
    xmax = 10
    NA = 0.95
    n = 129
    z = 4
    zetes = np.linspace(-z, z, n)
    print(zetes.min())

    # Computa les funcions radials axials
    res = compute_radial(NA, zetes, n=1204)

    # Fes-ne un plot
    irradiance = np.sum(np.real(np.conj(res[:, :2])*res[:, :2]), axis=1)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].plot(zetes, irradiance-irradiance.min())
    ax[0].set_ylim(0, irradiance.max())
    ax[1].plot(zetes, abs(res[:, 2])**2)
    #ax[1].set_ylim(0, abs(res[:, 2]**2).max())
    ax[1].set_ylim(0, irradiance.max())

    fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
    ax2.plot(zetes, compute_dop(res))
    ax2.set_ylim(0, 1)
    ax2.set_title("DoP Radial")

    np.save("irr_azimuthal.npy", abs(res[:, 2])**2)

    np.savez("radial_res_int.npz", irr=irradiance, dop=compute_dop(res))

    plt.show()

if __name__ == "__main__":
    main()

