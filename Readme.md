# Axion Like Particle (ALP) photon mixing simulation

This package simulates mixing of ALPs into photons and its effect of gamma ray spectra of pulsars. This is the proof-of-concept and is not the final version of the code used for publication - Reconciling hints on axion-like-particles from high-energy gamma rays with stellar bounds. That paper (Section 2.2 and the appendix) uses a similar code but with different parameter values and configurations, as explained in the manuscript.

The code is split into two parts (1) The heavy ALP mixing simulation is performed in C++ and compiled into shared object file (comb_b_mod.so). (2) The .so file is called within python similar to a package and used to fast simulations. This is then used to explore (m,g) parameter space and get best-fit values / CI contours.

The package uses 400 simulations of each pulsar spectra (explained in manuscript). The simulated spectra is not provided here, but the base raw data is provided. The magnetic field params are nuisance parameters.

pyymw16 electron density model is used. However this can be replaced with just ne~0.1-1 and the result does not change a lot.

The code in python is parallelized, which further reduces runtime.

Within archive/ folder you see older versions of the same code, entire written in Python.
