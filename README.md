rednoise
========

run aia/py/aia/prep_aia_data.py

 - reads in a directory of fits files, solar de-rotates them, co-aligns them
   and then spits out cutouts of these data for further analysis

run aia/py/aia/aia_lstsqr4.py

 - reads in the cutouts, normalizes the time series, apodizes the time series,
   calculates cross correlation coefficients, calculates various measures of
   the power spectra, including the geometric mean

run aia/py/aia/aia_pymc_powerlaws3.py

 - reads in the geometric mean power spectrum, estimated errors, calculates
   the effective number of pixels, fits two models of the power spectrum,
   and calculates various fitness criteria.
