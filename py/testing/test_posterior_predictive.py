"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np
import os
from rnfit2 import Do_MCMC
from rnsimulation import TimeSeries, SimplePowerLawSpectrum, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from pymcmodels import single_power_law_with_constant


# Create some fake data
dt = 12.0
nt = 300
pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0], nt=nt, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls1).sample

ts = TimeSeries(dt * np.arange(0, nt), data)

filename = os.path.expanduser('~/Desktop/test.pickle')

# Analyze using MCMC
result = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)


# Do the posterior predictive statistics to measure GOF
def vaughan_2010_T_R(iobs, S):
    """Vaughan, 2010, MNRAS, 402, 307. Eq. 15.
    Returns a test statistic measuring the difference between
    the observed power spectrum iobs and another power spectrum S."""
    return np.max(2 * iobs / S)


def vaughan_2010_T_SSE(iobs, S):
    """Vaughan, 2010, MNRAS, 402, 307. Eq. 21.
    Returns a test statistic measuring the difference between
    the observed power spectrum iobs and another power spectrum S."""
    return np.sum(((iobs - S) / S) ** 2)


def vaughan_2010_T_LRT(logp_model1, logp_model2):
    """Vaughan, 2010, MNRAS, 402, 307. Eq. 22.
    Returns a test statistic used to compare nested models."""
    return -2 * (logp_model1 - logp_model2)


def posterior_predictive_distribution(iobs, fit_results,
                                      nsample=1,
                                      statistic='vaughan_2010_T_R',
                                      ):
    # Storage for the distribution results
    distribution = []

    # Sample from the Bayes posterior for the fit, generate a spectrum,
    # and calculate the test statistic
    for i in range(0, nsample):
        # get a random sample from the posterior
        r = np.random.randint(0, fit_results["power_law_index"].shape[0])
        norm = fit_results["power_law_norm"][r]
        index = fit_results["power_law_index"][r]
        background = fit_results["background"][r]

        # Define some simulated time series data with the required spectral
        # properties
        pls = SimplePowerLawSpectrumWithConstantBackground([norm, index, background], nt=nt, dt=dt)
        data = TimeSeriesFromPowerSpectrum(pls).sample

        # Create a TimeSeries object from the simulated data
        ts = TimeSeries(dt * np.arange(0, nt), data)

        # Get the simulated data's power spectrum
        S = ts.PowerSpectrum.ppower

        if statistic == 'vaughan_2010_T_R':
            value = vaughan_2010_T_R(iobs, S)
        if statistic == 'vaughan_2010_T_SSE':
            value = vaughan_2010_T_R(iobs, S)
        distribution.append(value)
    return np.array(distribution)

fit_results = result.results[0]["samples"]

x = posterior_predictive_distribution(ts.PowerSpectrum.ppower, fit_results)

value = vaughan_2010_T_R(ts.PowerSpectrum.ppower/ts.PowerSpectrum.ppower[0], pls1.power()/pls1.power()[0])
