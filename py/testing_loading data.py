"""
Testing RN
"""
 
import numpy as np
import matplotlib.pyplot as plt
import rn_utils
import pymc

# Set up the simulated data
n = 300
dt = 12.0
alpha = 3.0

directory = '/Users/ireland/ts/img/rednoise/simulated/alpha='+str(alpha)
format = 'png'

for seed in range(1,5):
    print seed
    # get some test data
    test_data = rn_utils.simulated_power_law(n, dt, alpha, seed = seed)
    
    # save the time-series to an image
    plt.figure(seed+20)
    plt.plot(test_data)
    plt.xlabel('sample number')
    plt.ylabel('emission')
    plt.title('simulated data alpha='+str(alpha))
    plt.savefig(directory+'_'+str(seed)+'_test_time_series.'+format, format=format)
    
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

    # plot the power spectrum and the quick fit
    plt.figure(seed + 30)
    plt.loglog(analysis_frequencies, analysis_power, label='observed power; true index: '+str(alpha))
    plt.loglog(analysis_frequencies, power_fit, label='fit, power law index: ' + str(m_estimate))
    plt.xlabel('frequency')
    plt.ylabel('power')
    plt.title('Data fit with simple single power law')
    plt.legend()
    plt.savefig(directory+'_'+str(seed)+'_power_spectrum.'+format, format=format)
    
    # PyMC definitions
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   value=m_estimate,
                                   lower=-1.0,
                                   upper=m_estimate + 2,
                                   doc='power law index')
    
    power_law_norm = pymc.Uniform('power_law_norm',
                                  value=c_estimate,
                                  lower=c_estimate * 0.8,
                                  upper=c_estimate * 1.2,
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
    M1 = pymc.MCMC([power_law_index, power_law_norm, power_law_spectrum, spectrum])
    
    # Run the sampler
    M1.sample(iter=50000, burn=1000, thin=10)
    
    # Get the power law index and save the results
    pli = M1.trace("power_law_index")[:]
    plt.figure(seed)
    plt.hist(pli, bins=40)
    plt.xlabel('power law index (true value='+str(alpha)+')')
    plt.ylabel('number found')
    plt.title('Probability distribution of power law index')
    plt.savefig(directory+'_'+str(seed)+'_test_power_law_index.'+format, format=format)


