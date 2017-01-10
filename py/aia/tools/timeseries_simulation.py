"""
The purpose of this module is to simulate time series from model power spectra.
The
"""

import numpy as np
from scipy.stats import chi2, uniform


class TimeSeriesFromModelSpectrum:
    def __init__(self, spectrum_model, a, nt=1800, dt=12.0, v=5, w=7):
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
        spectrum_model : a spectrum_model object from rnspectralmodels3
            defines the properties of the power spectrum

        a : numpy array
            parameter values of the spectral model

        nt : int
            number of samples in the final time series

        dt : float
            sample cadence (nominally in seconds)

        v : scalar >= 1
            oversampling parameter mimicking increasing the time series duration
            from N*dt to V*N*dt

        w : scalar >= 1
            oversampling parameter mimicking increasing the cadence from dt to
            dt/W

        Examples
        --------


        """
        # Model Fourier power spectrum
        self.spectrum_model = spectrum_model

        # Parameters of the spectral model
        self.a = a

        # Number of elements in the time series generated
        self.nt = nt

        # Sample cadence of the final time series.
        self.dt = dt

        # Vaughan (2010) oversampling parameters
        self.v = v
        self.w = w

        #
        self.k = self.v * self.w * self.nt / 2

        # Frequencies that we are calculating the power spectrum at
        self.frequencies = np.arange(1, self.k + 1) / (1.0*(self.v * self.nt * self.dt))

        # Non-noisy power at the frequencies we are interested in
        self.power = self.spectrum_model(self.a, self.frequencies)

    def sample(self, fft_zero=0.0, phase_noise=True, power_noise=True):
        """
        Return a time-series with the given properties

        :param fft_zero:
        :param phase_noise:
        :param power_noise:
        :return:
        """

        # The fully over-sampled timeseries, with a sampling cadence of dt/W,
        # and a duration of V*N*dt.
        oversampled = time_series_from_power_spectrum(self.power,
                                                      fft_zero=fft_zero,
                                                      phase_noise=phase_noise,
                                                      power_noise=power_noise)

        # Subsample the time-series back down to the requested cadence of dt
        long_timeseries = oversampled[0:len(oversampled):self.w]
        nlts = len(long_timeseries)

        # Get a sample of the desired length nt from the middle of the long
        # time series.
        return self.dt * np.arange(0, self.nt),\
               long_timeseries[nlts//2 - self.nt//2:
                               nlts//2 - self.nt//2 + self.nt]


def time_series_from_power_spectrum(power_spectrum, fft_zero=0.0,
                                    phase_noise=True, power_noise=True):
    """Create a time series with power law noise, following the recipe
    of Vaughan (2010), MNRAS, 402, 307, appendix B

    Parameters
    ----------
    power_spectrum : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.

    fft_zero : scalar number
        The value at the zero Fourier frequency.
    """

    # Apply noise to the Fourier power spectrum
    if power_noise:
        fps = noisy_power_spectrum(power_spectrum)
    else:
        fps = power_spectrum

    # Get noisy Fourier transform
    f_complete, f = noisy_fourier_transform(fps,
                                            fft_zero=fft_zero,
                                            phase_noise=phase_noise)

    # Create the time-series.  The complex part should be tiny.
    # T_sim = np.fft.irfft(np.concatenate((np.asarray([fft_zero]), F)))

    # The time series is formally complex.  Return the real part only.
    # return np.real(T_sim)
    return np.fft.ifft(f_complete)


def noisy_power_spectrum(power_spectrum):
    """
    Create a noisy power spectrum, given some input power spectrum S, following
    the recipe of Vaughan (2010), MNRAS, 402, 307, appendix B

    Parameters
    ----------
    power_spectrum : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.

    """
    # Number of positive frequencies to calculate.
    k = len(power_spectrum)

    # chi-squared(2) random numbers for all frequencies except the nyquist
    # frequency
    x = np.concatenate((chi2.rvs(2, size=k-1), chi2.rvs(1, size=1)))

    # power spectrum
    return power_spectrum * x / 2.0


def noisy_fourier_transform(power_spectrum, fft_zero=0.0, phase_noise=True):
    """
    Take an input power spectrum I and create a full Fourier transform over all
    positive and negative frequencies and has random phases such that the time
    series it describes is purely real valued
    """
    # Number of positive frequencies to calculate.  This fills in the Fourier
    # frequency a[1:n/2+1].  Since we are working with the power spectrum to
    # generate a time-series, the resulting time-series is always even.
    k = len(power_spectrum)

    # random phases, except for the nyquist frequency.
    ph = uniform.rvs(loc=-np.pi / 2.0, scale=np.pi, size=k)
    if not phase_noise:
        ph[:] = 0.0
    ph[-1] = 0.0

    # Amplitudes
    a = np.sqrt(power_spectrum / 2.0)

    # print np.sqrt(I)

    # WARNING - SPECTRAL POWER APPEARs TO BE OUT BY A FACTOR 2 WITHOUT THIS
    # MULTIPLICATION FACTOR BELOW
    a = a * np.sqrt(2.0)

    # Complex vector
    f = a * np.exp(-np.complex(0, 1) * ph)

    # Form the negative frequency part
    f_negative = np.conjugate(f)[::-1][1:]

    # Form the fourier transform
    f_complete = np.concatenate((np.asarray([fft_zero]), f, f_negative))

    return f_complete, f
