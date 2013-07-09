"""
Red noise simulation functions
"""

import numpy as np
import rnspectralmodels
from scipy.stats import chi2, uniform


class PowerLawPowerSpectrum:
    def __init__(self, parameters, frequencies=None, nt=300, dt=12.0):
        # Positive frequencies at which the power spectrum is calculated
        if frequencies is None:
            self.nt = nt
            self.dt = dt
            self.frequencies = equally_spaced_nonzero_frequencies(self.nt, self.dt)
        else:
            self.frequencies = frequencies
            self.nt = len(self.frequencies)
            self.dt = 1.0/(self.frequencies[0]) - 1.0/(self.frequencies[1])

        self.parameters = parameters

class SimplePowerLawSpectrum(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.power_law(self.frequencies, self.parameters)

class SimplePowerLawSpectrumWithConstantBackground(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.power_law_with_constant(self.frequencies, self.parameters)

class BrokenPowerLaw(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.broken_power_law(self.frequencies, self.parameters)


class TimeSeriesFromPowerSpectrum():
    """
    Create a time series with a given type of power spectrum
    """
    def __init__(self, powerspectrum, V=10, W=10, seed=None):
        self.powerspectrum = powerspectrum
        self.nt = self.powerspectrum.nt

        # Vaughan (2010)
        self.V = V
        self.W = W
        
        # Random number seed
        self.seed = seed
        
        #
        self.K = self.V * self.W * self.nt / 2
        self.dt = self.powerspectrum.dt
        
        # Frequencies that we are calculating the power spectrum at
        self.frequencies = np.arange(1, self.K + 1)/(self.V * self.nt * self.dt)
        self.powerspectrum.frequencies=self.frequencies
        self.power = self.powerspectrum.power()

        # the fully over-sampled timeseries, with a sampling cadence of dt/W, with a
        # duration of V*N*dt
        self.oversampled = power_law_noise_random_phases(self.power, seed=self.seed)

        # Subsample the time-series back down to the requested cadence of dt
        self.longtimeseries = self.oversampled[np.arange(0, len(self.oversampled), self.W)]
        nlts = len(self.longtimeseries)

        # get a sample of the desired length nt from the middle of the long time series
        self.sample = self.longtimeseries[nlts/2-self.nt/2: nlts/2-self.nt/2 + self.nt]

def equally_spaced_nonzero_frequencies(n, dt):
    """
    Create a set of equally spaced Fourier frequencies
    """
    all_fft = np.fft.fftfreq(n, d=dt)
    return all_fft[all_fft>0]

def power_law_noise_random_phases(S, fft_zero=0.0, seed=None):
    """Create a time series with power law noise, following the recipe
    of Vaughan (2010), MNRAS, 402, 307, appendix B
    
    Parameters
    ----------
    S : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.
 
    fft_zero : scalar number
        The value at the zero Fourier frequency.

    seed : scalar number
        A seed value for the random number generator
     
    """
    
    # Number of positive frequencies to calculate.  This fills in the Fourier
    # frequency a[1:n/2+1].  Since we are working with the power spectrum to
    # generate a time-series, the resulting time-series is always even.
    K = len(S)

    # chi-squared(2) random numbers for all frequencies except the nyquist
    # frequency
    X = np.concatenate((chi2.rvs(2, size=K-1, seed=seed),
                        chi2.rvs(1, size=1, seed=seed)))
    
    # random phases, except for the nyquist frequency.
    ph = uniform.rvs(loc=-np.pi/2.0, scale=np.pi, size=K, seed=seed)
    ph[-1] = 0.0
    
    # power spectrum
    I = S*X/2.0
    
    # Amplitudes
    A = np.sqrt(I/2.0)
    
    # Complex vector
    F = A*np.exp(-np.complex(0,1)*ph)
    
    # Form the negative frequency part
    F_negative = np.conjugate(F)[::-1][1:]
    
    # Form the fourier transform
    F_complete = np.concatenate((np.asarray([fft_zero]), F, F_negative))
    
    # create the time-series.  The complex part should be tiny.
    T_sim = np.fft.ifft(F_complete)
    
    # The time series is formally complex.  Return the real part only.
    return np.real(T_sim)