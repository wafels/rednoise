"""
The purpose of this program is analyze test data so we can find out how long
a time series has to be in order for it to have enough information in it that
we are able to determine its power law index a reasonable percentage of the
time. 
"""
 
import numpy as np
import matplotlib.pyplot as plt
import rn_utils
import pymc 

# Set up the simulated data

# Initial length of the time series
n_initial = 300

# Sampling (SDO AIA sample rate is once every 12 seconds)
dt = 12.0

# power law index
alpha = 2.0

# The time-series is increased by this number of samples
n_increment = 300

# Maximum number of increments
max_increment = 12

# Number of trial data runs at each time-series length
ntrial = 1000

# PyMC variables
iterations = 50000
burn = iterations / 4
thin = 5

# storage array for the credible interval
ci_keep68 = np.zeros((max_increment, ntrial, 2))

# Least squares fit found
lstsqr_fit = np.zeros((max_increment, ntrial, 2))

# Bayesian posterior means
bayes_mean = np.zeros((max_increment, ntrial, 2))

# Storage array for the fraction found
fraction_found = np.zeros((max_increment))

for i in range(0, max_increment):

    # increase the length of the time series
    n = n_initial + i * n_increment

    for j in range(0, ntrial):
        print j
        # ensure repeatable random time-series.  We will need this when we
        # come to analyze the same data in different ways
        seed = j + i * ntrial
        test_data = rn_utils.simulated_power_law(n, dt, alpha, seed=seed)

        # get the power spectrum and frequencies we will analyze at
        observed_power_spectrum = (np.absolute(np.fft.fft(test_data))) ** 2
        fftfreq = np.fft.fftfreq(n, dt)
        analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
        analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

        # get a simple estimate of the power law index
        estimate = rn_utils.do_simple_fit(analysis_frequencies, analysis_power)
        c_estimate = estimate[0]
        m_estimate = estimate[1]
        print m_estimate

        # Keep the least squares fit
        lstsqr_fit[i, j, 0] = c_estimate
        lstsqr_fit[i, j, 1] = m_estimate

        # plot the power law spectrum
        power_fit = c_estimate * analysis_frequencies ** (-m_estimate)

        # PyMC definitions
        # Define data and stochastics
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=-1.0,
                                       upper=m_estimate + 2,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=c_estimate * 0.01,
                                      upper=c_estimate * 100.0,
                                      doc='power law normalization')

        # Model for the power law spectrum
        @pymc.deterministic(plot=False)
        def power_law_spectrum(p=power_law_index,
                               a=power_law_norm,
                               f=analysis_frequencies):
            """A pure and simple power law model"""
            out = a * (f ** (-p))
            return out

        spectrum = pymc.Exponential('spectrum',
                               beta=1.0 / power_law_spectrum,
                               value=analysis_power,
                               observed=True)

        # Set up the MCMC model
        M1 = pymc.MCMC([power_law_index,
                        power_law_norm,
                        power_law_spectrum,
                        spectrum])

        # Run the sampler
        M1.sample(iter=iterations, burn=burn, thin=thin)

        # Get the power law index and the normalization
        pli = M1.trace("power_law_index")[:]
        m_posterior_mean = np.mean(pli)
        cli = M1.trace("power_law_norm")[:]
        c_posterior_mean = np.mean(cli)

        bayes_mean[i, j, 0] = c_posterior_mean
        bayes_mean[i, j, 1] = m_posterior_mean

        # Get various properties of the power law index marginal distribution
        # and save them
        ci_keep68[i, j, :] = rn_utils.credible_interval(pli, ci=0.68)

        # Plots
        # Calculate the actual fit
        bayes_fit = c_posterior_mean * analysis_frequencies ** (-m_posterior_mean)

        # plot the power spectrum, the quick fit, and the Bayesian fit
        plt.figure(1)
        plt.loglog(analysis_frequencies, analysis_power, label='observed power; true index: '+str(alpha))
        plt.loglog(analysis_frequencies, power_fit, label='lstsqr fit, power law index: ' + str(m_estimate))
        plt.loglog(analysis_frequencies, bayes_fit, label='posterior mean, power law index: ' + str(m_posterior_mean))
        plt.xlabel('frequency')
        plt.ylabel('power')
        plt.title('Data fit with simple single power law')
        plt.legend()

        plt.figure(2)
        plt.hist(pli, bins=40, label='68% CI=' + str(ci_keep68[i, j, 0]) + ',' + str(ci_keep68[i, j, 1]) )
        plt.title('power law index PDF')
        plt.legend()

        plt.figure(3)
        plt.hist(cli, bins=40)
        plt.title('power law normalization PDF')
        plt.show()

    # Calculate how often the credible interval includes the true value of the
    # power law index.
    low = np.squeeze(ci_keep68[i, :, 0]) <= alpha
    high = np.squeeze(ci_keep68[i, :, 1]) >= alpha
    fraction_found[i] = np.sum(low * high) / (1.0 * ntrial)
