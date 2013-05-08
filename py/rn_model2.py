"""
A PyMC model for observed spectra.
"""

import pymc as pymc
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.io import readsav

__all__ = ['analysis_power', 'analysis_frequencies', 'power_law_index',
           'power_law_norm', 'power_law_spectrum', 'spectrum']

# data goes here
#input_power_spectrum = read in from file???

def main():

    def rednoise_test(n, dt, alpha):
        wn = np.random.normal(size=(n,))
    
        # FFT of the white noise
        wn_fft = np.fft.rfft(wn)
    
        # frequencies
        f = np.fft.fftfreq(n, dt)[:len(wn_fft)]
        f[-1]=np.abs(f[-1])
    
        fft_sim = wn_fft[1:] * f[1:]**-alpha
        T_sim = np.fft.irfft(fft_sim)
        return T_sim
    
    def testdata(n):
        return np.random.normal(size=(n))
    
    
    def read_idl(fullpath):
        s = readsav(fullpath)
        data = np.sum(np.sum(s.region_window, axis=1, dtype=np.float64), axis=1, dtype=np.float64)
        dt = 12.0
        fftfreq = np.fft.fftfreq(len(data), dt)
        return fftfreq, data

    
    n = 300
    dt = 12.0
    test_data = rednoise_test(n+2, dt, 1.0)
    
    #fftfreq, test_data = read_idl('/home/ireland/Data/oscillations/mcateer/outgoing3/AR_B.sav')
    
    input_power_spectrum = abs(np.fft.fft(test_data)) ** 2
    
    """
    (1) Conditions on the frequencies...
    
    Drop the zero frequency and the nyquist frequency since these do
    not have the exponential distribution.
    
    (2) Conditions on the observed powers
    
    Need to drop the observed power at the zero and nyquist frequencies, since
    those frequencies have been dropped.
    
    Would also be good to normalize the power spectrum in some way so that priors
    can be more independent of the data.  Note that this can also be handled by
    adjusting the prior, rather than normalizing the data.
    """
    
    fftfreq = np.fft.fftfreq(n, dt)
    analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
    analysis_power = input_power_spectrum[fftfreq >= 0][1:-1]
    
    # get a quick estimate assuming the data is a power law only.
    # the [0] entry from lstsq is the gradient, the [1] entry is the
    # value at np.log(observed_power_spectrum) = 0.0
    coefficients = np.polyfit(np.log(analysis_frequencies),
                               np.log(analysis_power),
                               1)
    m_estimate = coefficients[0]
    c_estimate = np.exp(coefficients[1])
    print m_estimate, c_estimate
    
    plt.figure(1)
    plt.loglog(analysis_frequencies, analysis_power)
    plt.loglog(analysis_frequencies, c_estimate * (analysis_frequencies) ** m_estimate)
    plt.show()
    
    plt.figure(2)
    plt.plot(test_data)
    plt.show()
    
    # Define data and stochastics
    power_law_index = pymc.Uniform('power_law_index',
                                   value = -m_estimate,
                                   lower=0.0,
                                   upper=-m_estimate + 2,
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
        out = a / (f ** p)
        return out
    
    #@deterministic(plot=False)
    #def power_law_spectrum_with_constant(p=power_law_index, a=power_law_norm,
    #                                     c=constant, f=frequencies):
    #    """Simple power law with a constant"""
    #    out = empty(frequencies)
    #    out = c + a/(f**p)
    #    return out
    
    spectrum = pymc.Exponential('spectrum',
                           beta=power_law_spectrum,
                           value=analysis_power,
                           observed=True)
    return True


if __name__ == '__main__':
    main()
