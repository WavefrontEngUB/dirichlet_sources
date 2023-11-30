#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def get_names(mode):
    fnames = os.listdir(f"{mode}/npys")
    dops = []
    irrs = []
    for name in fnames:
        if name.endswith("dop.npy"):
            dops.append(name)
        elif name.endswith("irr.npy"):
            irrs.append(name)
    dops.sort()
    irrs.sort()
    return dops, irrs

def save_video(name, stack):
    writer = imageio.get_writer(name, fps=15)
    
    images = np.uint8(stack/stack.max()*255)
    ny, nx, nz = stack.shape
    for i in range(nz):
        writer.append_data(images[:, :, i])
    writer.close()

def plot_mode(ax, dop, irr, mode, z=4):
    n = len(dop)
    ax[0].plot(np.linspace(-z, z, n), np.sqrt(dop))
    ax[0].set_ylim([0, 1.1])
    ax[0].set_title(f"DoP, N={mode}")

    ax[1].plot(np.linspace(-z, z, n), irr)
    ax[1].set_ylim([0, irr.max()*1.05])
    ax[1].set_title(f"Irradiance, N={mode}")

def main():
    import sys
    mode = sys.argv[1]
    dop_names, irr_names = get_names(mode)
    #try:
    #    os.mkdir("img_results")
    #except:
    #    pass

    for i in range(len(dop_names)):
        print(f"Processing {i}...")
        dop_n = dop_names[i]
        irr_n = irr_names[i]
        
        dop = np.load(f"{mode}/npys/{dop_n}")
        irr = np.load(f"{mode}/npys/{irr_n}")
        nx, ny, _ = dop.shape

        # Plotegem els resultats
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        plot_mode(ax, dop[0, 0, :], irr[0, 0, :], i)

        fig.savefig(f"{mode}/{i}-mode.png", dpi=200)
        fig.savefig(f"{mode}/{i}-mode.pdf", dpi=200)

        # Desem un vídeo de les irradiàncies
        save_video(f"{mode}/{i}-irr.mp4", np.fft.fftshift(irr, axes=(0, 1)))
        save_video(f"{mode}/{i}-dop.mp4", np.fft.fftshift(np.sqrt(dop), axes=(0, 1)))
        if i == len(dop_names)-1:
            np.savez(f"{mode}/axial_res_{mode}.npz", dop=dop[0, 0, :], irr=irr[0, 0, :])
        plt.close()

if __name__ == "__main__":
    main()
