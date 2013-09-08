import numpy as np
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum, noisy_power_spectrum


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



def posterior_predictive_distribution(iobs, fit_results, passed_mean, passed_std,
                                      nsample=1000,
                                      statistic='vaughan_2010_T_R',
                                      nt=300, dt=12.0, M=None
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
        data = TimeSeriesFromPowerSpectrum(pls, V=1, W=1).sample

        # Create a TimeSeries object from the simulated data
        ts = TimeSeries(dt * np.arange(0, nt), data)

        # Get the simulated data's power spectrum
        #S = ts.PowerSpectrum.Npower
        #Sroot = pls.power()
        #S = noisy_power_spectrum(Sroot)
        S = M.trace("predictive")[i]
        S = S / passed_mean
        S = S / passed_std
        # Calculate the required discrepancy statistic
        if statistic == 'vaughan_2010_T_R':
            value = vaughan_2010_T_R(iobs, S)
        if statistic == 'vaughan_2010_T_SSE':
            value = vaughan_2010_T_SSE(iobs, S)
        distribution.append(value)

    return np.array(distribution)