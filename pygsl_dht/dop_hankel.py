import numpy as np
import matplotlib.pyplot as plt
from pydht import DHT

def kernA(rho, z):
    sq = np.sqrt(1-rho*rho)
    k = np.exp(-1j*2*np.pi*z*sq)*sq*rho

    return k

def kernC(rho, z):
    sq = np.sqrt(1-rho*rho)
    k = np.exp(-1j*2*np.pi*z*sq)*rho*rho

    return k

def main():
    N = 256
    xmax = 4

    dht = DHT(N)
    dht.init(1, 4)

if __name__ == "__main__":
    main()
