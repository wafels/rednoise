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
bayes_mean = np.zeros((nenergy, 4))

# Bayesian posterior modes
bayes_mode = np.zeros((nenergy, 4))

# Bayesian posterior modes
ci_keep68 = np.zeros((nenergy, 3, 2))

for seed in range(0, 9):

    # get some test data
    data = z['obs_data'][:, seed][0:700]
    
    # get the length of the data
    n = len(data)

    # get the power spectrum and frequencies we will analyze at
    observed_power_spectrum = (np.absolute(np.fft.fft(data/np.std(data)))) ** 2
    fftfreq = np.fft.fftfreq(n, dt)
    analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
    analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

    # get a simple estimate of the power law index
    estimate = rn_utils.do_simple_fit(analysis_frequencies[0:n/5], analysis_power[0:n/5])
    c_estimate = estimate[0]
    m_estimate = estimate[1]
    b_estimate = np.log(np.mean(analysis_power[-n/1.5:-1]))
    f_estimate = np.log10( 0.5*np.max(analysis_frequencies))

    # Define the model we are going to use
    broken_power_law_model = \
    pymcmodels.broken_power_law(analysis_frequencies,
                                              analysis_power,
                                              c_estimate,
                                              m_estimate,
                                              b_estimate,
                                              f_estimate)
    # Set up the MCMC model
    M1 = pymc.MCMC(broken_power_law_model)
    
    # Run the sampler
    M1.sample(iter=50000, burn=10000, thin=10)

    # Get the power law index and the normalization
    d1 = M1.trace("delta1")[:]
    bayes_mean[seed, 0] = np.mean(d1)

    d2 = M1.trace("delta2")[:]
    bayes_mean[seed, 1] = np.mean(d2)
    
    norm = M1.trace("power_law_norm")[:]
    bayes_mean[seed, 2] = np.mean(norm)
    
    brk = M1.trace("breakf")[:]
    bayes_mean[seed, 3] = 10.0**np.mean(brk)

    # Get various properties of the power law index marginal distribution
    # and save them
    #ci_keep68[seed, 1, :] = rn_utils.credible_interval(pli, ci=0.68)
    #ci_keep68[seed, 0, :] = rn_utils.credible_interval(cli, ci=0.68)
    #ci_keep68[seed, 2, :] = rn_utils.credible_interval(bli, ci=0.68)

    bayes_mode[seed, 0] = rn_utils.bayes_mode(d1, bins, 0.1)
    bayes_mode[seed, 1] = rn_utils.bayes_mode(d2, bins, 0.2)
    bayes_mode[seed, 2] = rn_utils.bayes_mode(norm, bins, 0.2)
    bayes_mode[seed, 3] = rn_utils.bayes_mode(brk, bins, 0.2)

    # save the time-series to an image
    # plot the time series, the power spectrum with the Bayes mode fit overlaid
    # and probability distributions for the normalization, delta1, f0, delta2
    plt.figure(seed, figsize=(16, 12))

    plt.subplot(2, 3, 1)
    plt.plot(data)
    plt.xlabel('sample number')
    plt.ylabel('counts')
    plt.title(str(z['obs_energies'][seed][0]) + ' - ' + str(z['obs_energies'][seed][1]) + ' keV')

    plt.subplot(2, 3, 2)
    fit = np.zeros(shape=analysis_frequencies.shape)
    
    cond = analysis_frequencies<bayes_mean[seed, 3]
 
    fit[cond] = np.exp(bayes_mean[seed, 2]) * \
    (analysis_frequencies[cond] ** -bayes_mean[seed, 0])
    
    cond = analysis_frequencies>=bayes_mean[seed, 3]
    fit[cond] = np.exp(bayes_mean[seed, 2]) * \
    (analysis_frequencies[cond] ** -bayes_mean[seed, 1]) * (bayes_mean[seed, 3]) **(bayes_mean[seed, 1] - bayes_mean[seed, 0])

    
    plt.loglog(analysis_frequencies, analysis_power, label=r'observed power')
    plt.loglog(analysis_frequencies, fit, label=r'Bayes mode fit')
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('Data fit with single power law and constant')
    plt.legend(loc=3)

    # Get the power law index and save the results
    plt.subplot(2, 3, 3)
    plt.hist(d1, bins=40)
    plt.xlabel('delta1')
    plt.ylabel('number found')
    plt.title('Probability distribution of the delta1 power law index')

    # Get the power law index and save the results
    plt.subplot(2, 3, 4)
    plt.hist(d2, bins=40)
    plt.xlabel('delta2')
    plt.ylabel('number found')
    plt.title('Probability distribution of the delta2 power law index')

    # Get the power law index and save the results
    plt.subplot(2, 3, 5)
    plt.hist(norm, bins=40)
    plt.xlabel('normalization')
    plt.ylabel('number found')
    plt.title('Probability distribution of the normalization value')
    plt.savefig(directory + str(seed) +'.png')
    
    plt.subplot(2, 3, 6)
    plt.hist(brk, bins=40)
    plt.xlabel('break frequency')
    plt.ylabel('number found')
    plt.title('Probability distribution of the break frequency')
    plt.savefig(directory + str(seed) +'.png')

