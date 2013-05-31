"""
Testing RN
"""
 
import numpy as np
import matplotlib.pyplot as plt
import rn_utils, pymcmodels
import pymc
from scipy.io import readsav

# Set up the simulated data

dt = 4.0

directory = '/Users/ireland/ts/img/rednoise/fp'
format = 'png'

z = readsav('/Users/ireland/ts/sav/obs_summ_20110607.sav')
nenergy = 9

# Histogram bin size
bins = 40

# Bayesian posterior means
bayes_mean = np.zeros((nenergy, 3))

# Bayesian posterior modes
bayes_mode = np.zeros((nenergy, 3))

# Bayesian posterior modes
ci_keep68 = np.zeros((nenergy, 3, 2))

for seed in range(0, 9):

    # get some test data
    data = z['obs_data'][:, seed]
    
    # get the length of the data
    n = len(data)

    # get the power spectrum and frequencies we will analyze at
    observed_power_spectrum = (np.absolute(np.fft.fft(data/np.std(data)))) ** 2
    fftfreq = np.fft.fftfreq(n, dt)
    analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
    analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

    # get a simple estimate of the power law index
    estimate = rn_utils.do_simple_fit(analysis_frequencies, analysis_power)
    c_estimate = estimate[0]
    m_estimate = estimate[1]
    b_estimate = np.mean(analysis_power[-n/1.5:-1])

    # Define the model we are going to use
    single_power_law_model = \
    pymcmodels.single_power_law_with_constant(analysis_frequencies,
                                              analysis_power,
                                              c_estimate,
                                              m_estimate,
                                              b_estimate)
    # Set up the MCMC model
    M1 = pymc.MCMC(single_power_law_model)
    
    # Run the sampler
    M1.sample(iter=50000, burn=10000, thin=10)

    # Get the power law index and the normalization
    pli = M1.trace("power_law_index")[:]
    bayes_mean[seed, 1] = np.mean(pli)
    cli = M1.trace("power_law_norm")[:]
    bayes_mean[seed, 0] = np.mean(cli)
    bli = M1.trace("background")[:]
    bayes_mean[seed, 2] = np.mean(bli)

    # Get various properties of the power law index marginal distribution
    # and save them
    ci_keep68[seed, 1, :] = rn_utils.credible_interval(pli, ci=0.68)
    ci_keep68[seed, 0, :] = rn_utils.credible_interval(cli, ci=0.68)
    ci_keep68[seed, 2, :] = rn_utils.credible_interval(bli, ci=0.68)

    bayes_mode[seed, 1] = rn_utils.bayes_mode(pli, bins, 0.1)
    bayes_mode[seed, 0] = rn_utils.bayes_mode(cli, bins, 0.2)
    bayes_mode[seed, 2] = rn_utils.bayes_mode(bli, bins, 0.2)

    # save the time-series to an image
    # plot the time series, the power spectrum with the Bayes mode fit overlaid
    # and probability distributions for the normalization, delta1, f0, delta2
    plt.figure(seed, figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(data)
    plt.xlabel('sample number')
    plt.ylabel('counts')
    plt.title(str(z['obs_energies'][seed][0]) + ' - ' + str(z['obs_energies'][seed][1]) + ' keV')

    plt.subplot(2, 2, 2)
    fit = bayes_mean[seed, 0] * (analysis_frequencies ** -bayes_mean[seed, 1]) + bayes_mean[seed, 2]
    plt.loglog(analysis_frequencies, analysis_power, label=r'observed power')
    plt.loglog(analysis_frequencies, fit, label=r'Bayes mode fit')
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('Data fit with single power law and constant')
    plt.legend(loc=3)

    # Get the power law index and save the results
    plt.subplot(2, 2, 3)
    pli = M1.trace("power_law_index")[:]
    plt.hist(pli, bins=40)
    plt.xlabel('Power law index')
    plt.ylabel('number found')
    plt.title('Probability distribution of the delta1 power law index')

    # Get the power law index and save the results
    plt.subplot(2, 2, 4)
    pli = M1.trace("background")[:]
    plt.hist(bli, bins=40)
    plt.xlabel('background')
    plt.ylabel('number found')
    plt.title('Probability distribution of the background value')
    plt.savefig(directory + str(seed) +'.png')

