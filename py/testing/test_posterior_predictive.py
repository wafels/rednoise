"""
The purpose of this program is to analyze test data of a given power law index
and duration and find the probability distribution of measured power law
indices.
"""

import numpy as np
from rnfit2 import Do_MCMC
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from pymcmodels import single_power_law_with_constant
import ppcheck


# Create some fake data
dt = 12.0
nt = 300
pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0], nt=nt, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls1).sample
t = dt * np.arange(0, nt)
#data = data + 0.5 * data.max() * np.sin(2 * np.pi * t / 300.0)

# Create a time series object
ts = TimeSeries(t, data)

# Analyze using MCMC
analysis = Do_MCMC([ts]).okgo(single_power_law_with_constant, iter=50000, burn=10000, thin=5, progress_bar=False)


# Get the MCMC results from the analysis
fit_results = analysis.results[0]["samples"]

# Get the MAP values
mp = analysis.results[0]["mp"]

# Get the list of variable names
l = str(list(mp.variables)[0].__name__)

# Best fit spectrum
mean = analysis.results[0]["mean"]
std = analysis.results[0]["std"]
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()

# Normalized observed power spectrum
iobs = ts.PowerSpectrum.Npower

plt.figure(1)
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label='normalized simulated power spectrum')
plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label ='best fit')
plt.axvline(1.0 / 300.0, label='5 mins')
plt.axvline(1.0 / 180.0, label='3 mins')
plt.legend()
plt.show()

# Calculate the posterior predictive distribution
nsample = 5000
distribution, allpower = ppcheck.posterior_predictive_distribution(iobs, fit_results, nsample=nsample, nt=nt, dt=dt)

# Calculate the discrepancy statistic
# Calculate the discrepancy statistic
value = {}
value["vaughan_2010_T_R"] = ppcheck.vaughan_2010_T_R(iobs, best_fit_power_spectrum)
value["vaughan_2010_T_SSE"] = ppcheck.vaughan_2010_T_SSE(iobs, best_fit_power_spectrum)

for i, test_stat in enumerate(distribution):
    x = np.asarray(distribution[test_stat])
    v = value[test_stat]
    plt.figure(2 + i)
    plt.hist(x, bins=100, range=[x.min(), x.min() + 100 * v])
    plt.axvline(v, label='best fit')
    plt.xlabel('value of test statistic')
    plt.ylabel('number found (%i samples)' % nsample)
    plt.title(test_stat)
    plt.show()


