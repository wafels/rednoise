"""
Testing RN
"""
 
import numpy as np
import matplotlib.pyplot as plt
import rn_utils
import pymc
import os
import pymcmodels

# Set up the simulated data
n = 600
dt = 1.0
alpha = 2.0

directory = os.path.expanduser('~/ts/img/rednoise/simulated/alpha='+str(alpha))
format = 'png'

for seed in range(0,5):
    print seed
    # get some test data
    test_data = rn_utils.simulated_power_law(n, dt, alpha, seed=seed, minimum = 1.0, poisson=False, amplitude=10.0)
   
    # get the power spectrum and frequencies we will analyze at
    observed_power_spectrum = (np.absolute(np.fft.fft(test_data))) ** 2
    fftfreq = np.fft.fftfreq(n, dt)
    analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
    analysis_power = observed_power_spectrum[fftfreq >= 0][1:-1]
    
    # get a simple estimate of the power law index
    estimate = rn_utils.do_simple_fit(analysis_frequencies, analysis_power)
    c_estimate = estimate[0]
    m_estimate = estimate[1]
    
    # plot the power law spectrum
    power_fit = c_estimate * analysis_frequencies ** (-m_estimate)

    
    # Set up the MCMC model
    M1 = pymc.MCMC(pymcmodels.single_power_law(analysis_frequencies, analysis_power, m_estimate))
    
    # Run the sampler
    M1.sample(iter=50000, burn=1000, thin=10)
    
    # Get the power law index and save the results
    pli = M1.trace("power_law_index")[:]
    s_pli = rn_utils.summary_stats(pli, 0.05, bins=40)
    
    cli = M1.trace("power_law_norm")[:]
    s_cli = rn_utils.summary_stats(cli, 0.05, bins=40)
    
    bayes_mean_fit = np.exp(s_cli["mean"]) * (analysis_frequencies ** -s_pli["mean"])
    bayes_mode_fit = np.exp(s_cli["mode"]) * (analysis_frequencies ** -s_pli["mode"])

    # plot the power spectrum, the quick fit, and the Bayesian fit
    plt.figure(1)
    plt.plot(test_data)
    plt.xlabel('time (seconds); sample cadence = %4.2f second' % (dt))
    plt.ylabel('simulated emission')
    plt.title(r'Simulated data $\alpha_{true}$=%4.2f , seed=%8i ' % (alpha, seed))
    plt.legend()
    plt.savefig(directory + '_timeseries.'+ str(seed) +'.png', format=format)
    plt.close()

    plt.figure(2)
    plt.loglog(analysis_frequencies, analysis_power, label=r'observed power: $\alpha_{true}= %4.2f$' % (alpha))
    # plt.loglog(analysis_frequencies, power_fit, label=r'$\alpha_{lstsqr}=%4.2f$' % (m_estimate))
    plt.loglog(analysis_frequencies, bayes_mean_fit, label=r'$\overline{\alpha}=%4.2f$' % (s_pli["mean"]))
    plt.loglog(analysis_frequencies, bayes_mode_fit, label=r'$\alpha_{mode}=%4.2f$' % (s_pli["mode"]))
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('Data fit with simple single power law')
    plt.legend(loc=3)
    plt.savefig(directory + '_fourier_loglog.'+ str(seed) +'.png', format=format)
    plt.close()

    plt.figure(3)
    plt.loglog(analysis_frequencies, analysis_power/bayes_mean_fit, label=r'observed power / mean fit')
    plt.loglog(analysis_frequencies, analysis_power/bayes_mode_fit, label=r'observed power / mode fit')
    plt.axhline(y=1.0, color='black')
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('data / fit')
    plt.legend(loc=3)
    plt.savefig(directory + '_data_divided_by_fit.'+ str(seed) +'.png', format=format)
    plt.close()