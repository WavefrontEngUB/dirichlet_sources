import pydht
import numpy as np
import matplotlib.pyplot as plt

def circle(n, a):
    return np.float64(np.linspace(0, 1, n) < a)

def xy_map(radial_fun):
    """Mapeja una funció radial a xy"""
    n = radial_fun.shape[0]
    xy = np.zeros((n, n), dtype=np.float64)
    y, x = np.mgrid[-n//2:n//2, -n//2:n//2]
    # Mapeig radial
    xy = radial_fun[np.int_(np.sqrt(x*x+y*y))]

    return xy

def test(n=512, a=0.1):
    dht = pydht.DHT(n)
    dht.init(0., 1.)
    function = circle(n, a)

    result = dht.apply(function)
    print(result.shape)

    ksamples = dht.get_k_samples()

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    xy = xy_map(result)

    ax[0].plot(ksamples, result*result, ".")
    ax[1].imshow(xy*xy, cmap="hot")
    plt.show()

if __name__ == "__main__":
    n = 512
    a = 0.0035

    test(n=n, a=a)
