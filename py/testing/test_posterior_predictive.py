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
import ppcheck2


# -----------------------------------------------------------------------------
# Create some fake data
# -----------------------------------------------------------------------------
dt = 12.0
nt = 300
np.random.seed(seed=1)
pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0], nt=nt, dt=dt)
data = TimeSeriesFromPowerSpectrum(pls1).sample
t = dt * np.arange(0, nt)
data = data + 0.2 * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)

# Create a time series object
ts = TimeSeries(t, data)

# Get the normalized power and the positive frequencies
this = ([ts.PowerSpectrum.frequencies.positive, ts.PowerSpectrum.Npower],)


# -----------------------------------------------------------------------------
# Analyze using MCMC
# -----------------------------------------------------------------------------
analysis = Do_MCMC(this).okgo(single_power_law_with_constant,
                              iter=50000, burn=1000, thin=5,
                              progress_bar=False)

# Get the MCMC results from the analysis
fit_results = analysis.results[0]["samples"]

# Get the MAP values
mp = analysis.results[0]["mp"]

# Get the list of variable names
#l = str(list(mp.variables)[0].__name__)

# Best fit spectrum
best_fit_power_spectrum = SimplePowerLawSpectrumWithConstantBackground([mp.power_law_norm.value, mp.power_law_index.value, mp.background.value], nt=nt, dt=dt).power()

# Normalized observed power spectrum
iobs = ts.PowerSpectrum.Npower
print "Normalized power"
print iobs

# Value of the test statistic using the best fit power spectrum
value = ppcheck2.vaughan_2010_T_R(iobs, best_fit_power_spectrum)
print "value ", value

plt.figure(1)
plt.loglog(ts.PowerSpectrum.frequencies.positive, iobs, label='normalized simulated power spectrum')
plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum, label ='best fit')
plt.axvline(1.0 / 300.0, label='5 mins', color='k')
plt.axvline(1.0 / 180.0, label='3 mins', color='k')
plt.legend()
plt.show()


# -----------------------------------------------------------------------------
# Now do the posterior predictive check
# -----------------------------------------------------------------------------


# Storage for the distribution results
distribution = []

# Number of posterior samples
nposterior = fit_results["power_law_index"].shape[0]

# Number of samples taken from the posterior
nsample = 10

# PyMC object
M = analysis.results[0]["M"]
print("sample predictive")
M.trace("predictive")[0]
# Use the PyMC predictive to generate power series taken from the posterior
for i in range(0, nsample):
    # How many samples have we worked on?
    print('Sample number %i out of %i' % (i + 1, nsample))

    # get a random sample from the posterior
    r = np.random.randint(0, nposterior)
    print r

    # Get a posterior power spectrum
    S = M.trace("predictive")[r]
    print S

    # Normalize
    S = S / ts.PowerSpectrum.vaughan_mean
    S = S / ts.PowerSpectrum.vaughan_std

    # Generate the input for the MCMC algorithm
    this2 = ([ts.PowerSpectrum.frequencies.positive, S],)

    # Analyze using MCMC
    analysis2 = Do_MCMC(this2).okgo(single_power_law_with_constant,
                                    iter=50000, burn=1000, thin=5,
                                    progress_bar=False)

    # Get the MCMC results from the analysis
    fit_results2 = analysis2.results[0]["samples"]

    # Get the MAP values
    mp2 = analysis2.results[0]["mp"]

    # Best fit spectrum
    best_fit_power_spectrum2 = SimplePowerLawSpectrumWithConstantBackground([mp2.power_law_norm.value,
                                                                             mp2.power_law_index.value,
                                                                             mp2.background.value],
                                                                            nt=nt, dt=dt).power()

    # Value of the test statistic using the best fit power spectrum
    distribution.append(ppcheck2.vaughan_2010_T_R(S, best_fit_power_spectrum2))
