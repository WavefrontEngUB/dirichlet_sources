# dirichlet_sources
Set of programs used to compute the focal irradiances and degrees of coherence of partially coherent dirichlet sources.
The code is divided into two folders, following two different methods. The main one, `fft_method` computes the focal
polarization matrices using a dedicated fortran program, plotting the final results through python and matplotlib. The
second one, used mainly to compute the degree of coherence at the focal plane of a highly focusing optical
system, `pygsl_dht`, consists mostly of python code wrapping the GNU Scientific Library.

## fft_method
To compile the program used to compute the compute the polarization matrices at different planes near the focal
plane of a highly focusing optical system is done via the following command
```
make dirichlet
```
which, upong completion, creates an executable called `dirichlet`inside the `prg` folder. After that, to compute
the field between two planes separated by a distance `2z` centered around `z = 0`, we just need to type
```
./dirichlet [N_lateral] [N_modes] [z] [l_focal] [NA]
```
where
    - N_lateral: lateral number of sampling points
    - N_modes: lateral number of sampling points
    - n_z: number of z planes to compute
    - z: maximum distance from focus
    - l_focal: Half size of the focal region window
    - NA: Numerical aperture
    - mode: Polarization mode, radial or azimuthal (optional, defaults radial)

## pygsl_dht
First, we need to build the wrapping library using the script `setup.py`. After that, the degree of coherence
at `z = 0` can be computed and controlled using the script `focal_degree_coherence.py`.
