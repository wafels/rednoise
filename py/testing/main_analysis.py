from __future__ import absolute_import
"""
Main analysis module for Bayesian MCMC Fourier spectrum fitting
"""

import numpy as np
import sys
#import os
from matplotlib import pyplot as plt
#import sunpy
#import pymc
#import tsutils
from rnfit2 import Do_MCMC, rnsave
import ppcheck2
from pymcmodels import single_power_law_with_constant
#from cubetools import get_datacube
from timeseries import TimeSeries
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
#import pickle

def main_analysis(ts,ppcheck=True,plots=False,nsample=10):


    # _____________________________________________________________________________
    # Set up where to save the data, and the file type/
    save = rnsave(root='~/ts/pickle', description='simulated', filetype='pickle')

    #get the Fourier spectrum out of the timeseries object
    nt = ts.SampleTimes.nt
    dt = ts.SampleTimes.dt
    iobs = ts.PowerSpectrum.Npower
    this = ([ts.PowerSpectrum.frequencies.positive, iobs],)

    # Analyze using MCMC
    # -----------------------------------------------------------------------------
    analysis = Do_MCMC(this).okgo(single_power_law_with_constant,
                              iter=50000,
                              burn=10000,
                              thin=5,
                              progress_bar=False,
                              db=save.filetype,
                              dbname=save.MCMC_filename)

    # Get the MAP values
    mp = analysis.results[0]["mp"]

    # Get the full MCMC object
    #M = analysis.results.M
    M = analysis.M
    # Save the MCMC output using the recommended save functionality
    save.PyMC_MCMC(M)

    # Save everything else
    datatype = 'analysis_summary'
    #save.analysis_summary(analysis)


    # Get the list of variable names
    #l = str(list(mp.variables)[0].__name__)

    # Best fit spectrum
    best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()
    print mp.power_law_norm.value, mp.power_law_index.value, mp.background.value

    if ppcheck:
        # -----------------------------------------------------------------------------
        # Now do the posterior predictive check - computationally time-consuming
        # -----------------------------------------------------------------------------
        statistic = ('vaughan_2010_T_R', 'vaughan_2010_T_SSE')
        #nsample = 10
        value = {}
        for k in statistic:
            value[k] = ppcheck2.calculate_statistic(k, iobs, best_fit_power_spectrum)

        print value

        distribution = ppcheck2.posterior_predictive_distribution(ts,
                                                          M,
                                                          nsample=nsample,
                                                          statistic=statistic,
                                                          verbose=True)
        # Save the test statistic distribution information
        save.posterior_predictive((value, distribution))


    if plots:
        # -----------------------------------------------------------------------------
        # Summary plots
        # -----------------------------------------------------------------------------

        # Plot the best fit
        plt.figure()
        plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label="normalized observed power spectrum")
        plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label="best fit")
        plt.axvline(1.0 / 300.0, color='k', linestyle='--', label='5 mins')
        plt.axvline(1.0 / 180.0, color='k', linestyle=':', label='3 mins')
        #plt.legend(fontsize=10, loc=3)
        plt.show()

        # Discrepancy statistics
        for i, k in enumerate(statistic):
            v = value[k]
            d = distribution[k]
            pvalue = np.sum(d > v) / (1.0 * nsample)

            plt.figure(i)
            h = plt.hist(d, bins=20)
            plt.axvline(v, color='k')
            plt.xlabel("statistic value")
            plt.ylabel("Number found (%i samples)" % (nsample))
            plt.title('Statistic: ' + k)
            plt.text(v, np.max(h[0]), "p = %f" % (pvalue))
        plt.show()


