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

directory = '/Users/ireland/ts/img/rednoise/simulated/alpha='+str(alpha)
format = 'png'

z = readsav('/Users/ireland/ts/sav/obs_summ_20110607.sav')

for seed in range(1,5):
    print seed
    # get some test data
    data = z['obs_data'][:, seed]
    
    # devide by the stabdard deviation
    data = data/np.std(data)
    
    # get the length of the data
    n = len(data)

    # get the power spectrum and frequencies we will analyze at
    observed_power_spectrum = (np.absolute(np.fft.fft(data))) ** 2
    fftfreq = np.fft.fftfreq(n, dt)
    analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
    analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]

    # Define the model we are going to use
    broken_power_law_model = pymcmodels.broken_power_law(analysis_frequencies, analysis_power,
                            1.5, 0.0, )
    # Set up the MCMC model
    M1 = pymc.MCMC(broken_power_law_model)
    
    # Run the sampler
    M1.sample(iter=50000, burn=1000, thin=10)

    # save the time-series to an image
    # plot the time series, the power spectrum with the Bayes mode fit overlaid
    # and probability distributions for the normalization, delta1, f0, delta2
    plt.figure(1)
    plt.subplot(2, 3, 1)

    plt.figure(seed+20)
    plt.plot(data)
    plt.xlabel('sample number')
    plt.ylabel('counts')
    plt.title('')

    plt.loglog(analysis_frequencies, analysis_power, label=r'observed power: $\alpha_{true}= %4.2f$' % (alpha))
    plt.loglog(analysis_frequencies, power_fit, label=r'$\alpha_{lstsqr}=%4.2f$' % (m_estimate))
    plt.loglog(analysis_frequencies, bayes_mean_fit, label=r'$\overline{\alpha}=%4.2f$' % (bayes_mean[i, j, 1]))
    plt.loglog(analysis_frequencies, bayes_mode_fit, label=r'$\alpha_{mode}=%4.2f$' % (bayes_mode[i, j, 1]))
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('Data fit with simple single power law')
    plt.legend(loc=3)

    # Get the power law index and save the results
    pli = M1.trace("delta1")[:]
    plt.figure(seed)
    plt.hist(pli, bins=40)
    plt.xlabel('delta1')
    plt.ylabel('number found')
    plt.title('Probability distribution of the delta1 power law index')

    # Get the power law index and save the results
    pli = M1.trace("delta2")[:]
    plt.figure(seed)
    plt.hist(pli, bins=40)
    plt.xlabel('delta2')
    plt.ylabel('number found')
    plt.title('Probability distribution of the delta2 power law index')


