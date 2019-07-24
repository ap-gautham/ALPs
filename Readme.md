Fit-DRM_2 file is slower but conversion probability function(P_ga) is easir to understand.

Fit_Final-ICRC file is much fasteer but P_ga function is jumbled up.

Fit_MultiNest is multinest implementation of Photon ALPs conversion fit.

To run the programs you'll need gmf.py from me-manu which can be found here - https://github.com/me-manu/gammaALPs/blob/master/gammaALPs/bfields/gmf.py

You'll aslo need minuit python package

To use multiprocessinng you'll need multiprocessing python  package. However one can modify the code to use it without this.

pyymw16 electron density model is used. However this can be replaced with just ne~0.1-1 and the result does not change a lot.
