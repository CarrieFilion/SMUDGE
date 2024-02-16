# SMUDGE
The Satellite Mergers Usher Disc Galaxy Evolution (SMUDGE) simulations are a set of four high-resolution, N-body dynamical simulations of disk galaxies with each simulation containing over a billion particles. This suite includes two different models of isolated, disk galaxies and two versions in which these same disk galaxies experience a minor merger with a satellite galaxy. These simulations were designed to be laboratories for the study of galactic dynamics rather than tailored models of the Milky Way, and the four available models enable the study of both the dynamics in isolated disks and the dynamics in merging, disequilibrium systems. 

* For full details of the simulations and the primary citation, see [Hunt et al 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1459H/abstract)
* The initial conditions for the host were taken from [Widrow & Dubinski 2005](https://ui.adsabs.harvard.edu/abs/2005ApJ...631..838W/abstract), and generated using Galactics [(Kuijken & Dubinski 1995)](https://ui.adsabs.harvard.edu/abs/1995MNRAS.277.1341K/abstract)
* The satellite is the L2 model from [Laporte et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481..286L/abstract) 
* The simulations were run with the GPU accelerated N-body tree code Bonsai [(Bédorf et al. 2012)](https://ui.adsabs.harvard.edu/abs/2012JCoPh.231.2825B/abstract)

The SMUDGE simulations are hosted on [SciServer](https://sciserver.org/datasets/) (see the website for details), and this code repo provides an example notebook and the package for interfacing with the simulations.

# Getting Started
If you don't already have one, make an account on [SciServer](https://sciserver.org/). From the Dashboard, click the Science Domains icon and then join the Cosmological Simulations Science Domain. To access the simulation snapshots, go to the Dashboard again and click the Compute icon and click 'Create a Container'. Select the SciServer 2.0 compute image, and then select the SMUDGE data volume. From there, you can clone this package in the terminal.

- note: you may have issues with importing coords from galpy.util - if this happens to you, run 'pip install galpy==1.7 --force-reinstall' at the top of your notebook or job, restart the kernel, and try again
  
## References
We ask that publications and presentations that use these simulations cite [Hunt et al 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1459H/abstract).
