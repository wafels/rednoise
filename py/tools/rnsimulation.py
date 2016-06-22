"""
Red noise simulation objects and functions
"""

import numpy as np
from scipy.stats import chi2, uniform


class SimulatedPowerSpectrum:
    def __init__(self, a, model):
        """
        Parent class of power law spectra

        Parameters
        ----------
        a : ndarray
            The parameters of the power law spectrum.
        model : func
            A Python function of the form power = func(a, f)

        """
        self.a = a
        self.model = model

    def power(self, f):
        """
        Return the power spectrum.  Placeholder for the methods defined below
        f : ndarray
            The frequencies at which to calculate the power
        """
        return self.model(self.a, f)

    def sample(self, f):
        """
        Return a noisy power spectrum sampled from the probability distribution
        of the noisy spectra
        """
        return noisy_power_spectrum(self.power(f))


class TimeSeriesFromSimulatedPowerSpectrum:
    def __init__(self, powerspectrum, f, fft_zero=0.0, **kwargs):
        """
        Create a time series with a given type of power spectrum.  This object
        can be used to generate time series that have a given power spectrum.
        The resulting time-series can be formed from oversamples of the source
        power spectrum.  This is done to better reproduce observed data.

        Observed power spectra from finite time-series have fixed sample
        frequencies given by the instrument sampling cadence.  The actual
        plasma has many more frequencies in it than are represented by any
        given time-series.  Therefore over-sampling - both at lower frequencies
        and at higher frequencies - is necessary to ensure that the simulated
        time-series better represents the actual underlying plasma.

        Parameters
        ----------
        powerspectrum : SimulatedPowerSpectrum object
            defines the properties of the power spectrum
        V : scalar >= 1
            oversampling parameter mimicking increasing the time series duration
            from N*dt to V*N*dt
        W : scalar >= 1
            oversampling parameter mimicking increasing the cadence from dt to
            dt/W

        fft_zero : scalar number
            The value at the zero Fourier frequency.

        """
        self.powerspectrum = powerspectrum
        self.nt = self.powerspectrum.nt
        self.fft_zero = fft_zero

        # Vaughan (2010)
        self.V = V
        self.W = W

        #
        self.K = self.V * self.W * self.nt / 2
        self.dt = self.powerspectrum.dt

        # Frequencies that we are calculating the power spectrum at
        self.frequencies = np.arange(1, self.K + 1) / (1.0*(self.V * self.nt * self.dt))
        self.powerspectrum.frequencies = self.frequencies
        self.inputpower = self.powerspectrum.power()

        # the fully over-sampled timeseries, with a sampling cadence of dt/W, with a
        # duration of V*N*dt
        self.oversampled = time_series_from_power_spectrum(self.inputpower, fft_zero=self.fft_zero, **kwargs)

        # Subsample the time-series back down to the requested cadence of dt
        self.longtimeseries = self.oversampled[0:len(self.oversampled):self.W]
        self.nlts = len(self.longtimeseries)

        # get a sample of the desired length nt from the middle of the long time series
        self.sample = self.longtimeseries[self.nlts / 2 - self.nt / 2: self.nlts / 2 - self.nt / 2 + self.nt]


def oversamplefrequencies(nt, dt, V, W):
    K = V * W * nt / 2
    return np.arange(1, K + 1) / (1.0 * (V * nt * dt))


def equally_spaced_nonzero_frequencies(n, dt):
    """
    Create a set of equally spaced Fourier frequencies
    """
    all_fft = np.fft.fftfreq(n, d=dt)
    return all_fft[all_fft > 0]


def noisy_power_spectrum(S):
    """
    Create a noisy power spectrum, given some input power spectrum S, following
    the recipe of Vaughan (2010), MNRAS, 402, 307, appendix B

    Parameters
    ----------
    S : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.

    """
    # Number of positive frequencies to calculate.
    K = len(S)

    # chi-squared(2) random numbers for all frequencies except the nyquist
    # frequency
    X = np.concatenate((chi2.rvs(2, size=K - 1),
                        chi2.rvs(1, size=1)))
    # power spectrum
    return S * X / 2.0


def noisy_fourier_transform(I, fft_zero=0.0, phase_noise=True):
    """
    Take an input power spectrum I and create a full Fourier transform over all
    positive and negative frequencies and has random phases such that the time
    series it describes is purely real valued
    """
    # Number of positive frequencies to calculate.  This fills in the Fourier
    # frequency a[1:n/2+1].  Since we are working with the power spectrum to
    # generate a time-series, the resulting time-series is always even.
    K = len(I)

    # random phases, except for the nyquist frequency.
    ph = uniform.rvs(loc=-np.pi / 2.0, scale=np.pi, size=K)
    if not phase_noise:
        ph[:] = 0.0
    ph[-1] = 0.0

    # Amplitudes
    A = np.sqrt(I / 2.0)

    #print np.sqrt(I)

    # WARNING - SPECTRAL POWER APPEARs TO BE OUT BY A FACTOR 2 WITHOUT THIS
    # MULTIPLICATION FACTOR BELOW
    A = A * np.sqrt(2.0)

    # Complex vector
    F = A * np.exp(-np.complex(0, 1) * ph)

    # Form the negative frequency part
    F_negative = np.conjugate(F)[::-1][1:]

    # Form the fourier transform
    F_complete = np.concatenate((np.asarray([fft_zero]), F, F_negative))

    return F_complete, F
    

def time_series_from_power_spectrum(S, fft_zero=0.0, phase_noise=True, power_noise=True):
    """Create a time series with power law noise, following the recipe
    of Vaughan (2010), MNRAS, 402, 307, appendix B

    Parameters
    ----------
    S : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.

    fft_zero : scalar number
        The value at the zero Fourier frequency.
    """

    # noisy power spectrum
    if power_noise:
        I = noisy_power_spectrum(S)
    else:
        I = S

    #print np.sqrt(I)

    # Get noisy Fourier transform
    F_complete, F = noisy_fourier_transform(I,
                                            fft_zero=fft_zero,
                                            phase_noise=phase_noise)

    # create the time-series.  The complex part should be tiny.
    T_sim = np.fft.ifft(F_complete)
    #T_sim = np.fft.irfft(np.concatenate((np.asarray([fft_zero]), F)))

    # The time series is formally complex.  Return the real part only.
    #return np.real(T_sim)
    return T_sim
