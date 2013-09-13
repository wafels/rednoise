import numpy as np
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground
from rnfit2 import Do_MCMC
from pymcmodels import single_power_law_with_constant
from matplotlib import pyplot as plt


# Calculate a statistic
def calculate_statistic(k, iobs, S):
    if k == 'vaughan_2010_T_R':
        return vaughan_2010_T_R(iobs, S)
    if k == 'vaughan_2010_T_SSE':
        return vaughan_2010_T_SSE(iobs, S)


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


def posterior_predictive_distribution(ts, M,
                                      nsample=1000,
                                      statistic=('vaughan_2010_T_R', 'vaughan_2010_T_SSE'),
                                      verbose=True):
    # Get some properties of the original time series
    nt = ts.SampleTimes.nt
    dt = ts.SampleTimes.dt

    # Storage for the distribution results
    distribution = {k: [] for k in statistic}

    # Number of posterior samples
    nposterior = M.trace("power_law_index")[:].size

    # Use the PyMC predictive to generate power series taken from the posterior
    for i in range(0, nsample):
        # How many samples have we worked on?
        if verbose:
            print('Sample number %i out of %i' % (i + 1, nsample))

        # get a random sample from the posterior
        r = np.random.randint(0, nposterior)

        # Get a posterior power spectrum
        S = M.trace("predictive")[r]

        # Normalize
        #S = S / ts.PowerSpectrum.vaughan_mean
        #S = S / ts.PowerSpectrum.vaughan_std

        # Generate the input for the MCMC algorithm
        this2 = ([ts.PowerSpectrum.frequencies.positive, S],)

        # Analyze using MCMC
        analysis2 = Do_MCMC(this2).okgo(single_power_law_with_constant,
                                 iter=50000, burn=1000, thin=5,
                                    progress_bar=False)

        # Get the MCMC results from the analysis
        #fit_results2 = analysis2.results[0]["samples"]

        # Get the MAP values
        mp2 = analysis2.results[0]["mp"]

        # Get the MAP values
        #mp2 = analysis2.results[0]["mp"]

        # Best fit spectrum
        best_fit_power_spectrum2 = SimplePowerLawSpectrumWithConstantBackground([mp2.power_law_norm.value,
                                                                                 mp2.power_law_index.value,
                                                                                 mp2.background.value],
                                                                                nt=nt, dt=dt).power()
        plt.subplot(3, 1, 1)
        plt.loglog(ts.PowerSpectrum.frequencies.positive, S)
        plt.loglog(ts.PowerSpectrum.frequencies.positive, best_fit_power_spectrum2)
        plt.subplot(3, 1, 2)
        plt.loglog(ts.PowerSpectrum.frequencies.positive, 2 * S / best_fit_power_spectrum2)
        plt.subplot(3, 1, 3)
        plt.loglog(ts.PowerSpectrum.frequencies.positive, ((S - best_fit_power_spectrum2) / best_fit_power_spectrum2)**2)
        plt.show()

        # Value of the test statistic using the best fit power spectrum
        for k in statistic:
            distribution[k].append(calculate_statistic(k,
                                                       S,
                                                       best_fit_power_spectrum2))
            print k, calculate_statistic(k, S, best_fit_power_spectrum2)
    return distribution
