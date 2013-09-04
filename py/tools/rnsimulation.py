"""
Red noise simulation functions
"""

import numpy as np
import rnspectralmodels
import copy
from scipy.stats import chi2, uniform


class SampleTimes:
    def __init__(self, time, label='time', units='seconds'):
        """A class holding time series sample times."""
        self.time = time - time[0]
        self.nt = self.time.size
        self.dt = self.time[-1] / (self.nt - 1)
        self.label = label
        self.units = units


#def cadence(t, absoluteTolerance=0.5):
    """#Get some information on the observed cadences in a SampleTimes object
    cadences = t.time[1:] - t.time[0:-1]
    segments = []
    iStart = 0
    iEnd = 0
    nsegment = 0
    repeat begin
        repeat begin
            iEnd = iEnd + 1
            c0 = cadence[iStart]
            c1 = cadence[iEnd]
        endrep until (abs(c1-c0) gt absoluteTolerance) or (iEnd eq nt-2)
        segment.append([iStart, iEnd])
        nsegment = nsegment + 1
        iStart = iEnd
    endrep until (iEnd eq nt-2)
"""
#    return None
#


class Frequencies:
    def __init__(self, frequencies, label='frequency', units='Hz'):
        self.frequencies = frequencies
        self.posindex = self.frequencies > 0
        self.positive = self.frequencies[self.posindex]
        self.label = label
        self.units = units


class PowerSpectrum:
    def __init__(self, frequencies, power, label='Fourier power'):
        self.frequencies = Frequencies(frequencies)
        self.power = power
        self.ppower = self.power[self.frequencies.posindex]
        self.label = label
        # normalized power
        d1 = self.ppower / np.mean(self.ppower)
        self.Npower = d1 / np.std(d1)


class TimeSeries:
    def __init__(self, time, data, label='data', units=''):
        self.SampleTimes = SampleTimes(time)
        if self.SampleTimes.nt != data.size:
            raise ValueError('length of sample times not the same as the data')
        self.data = data
        self.PowerSpectrum = PowerSpectrum(np.fft.fftfreq(self.SampleTimes.nt, self.SampleTimes.dt),
                                           np.abs(np.fft.fft(self.data)) ** 2)
        self.label = label
        self.units = units


class PowerLawPowerSpectrum:
    def __init__(self, parameters, frequencies=None, nt=300, dt=12.0):
        """
        Parent class of power law spectra

        Parameters
        ----------
        parameters : ndarray
            The parameters of the power law spectrum
        frequencies : ndarray
        """
        # Positive frequencies at which the power spectrum is calculated
        if frequencies is None:
            self.nt = nt
            self.dt = dt
            self.frequencies = equally_spaced_nonzero_frequencies(self.nt, self.dt)
        else:
            self.frequencies = frequencies
            self.nt = len(self.frequencies)
            self.dt = 1.0 / (self.frequencies[0]) - 1.0 / (self.frequencies[1])

        self.parameters = parameters

    def power(self):
        """
        Return the power spectrum.  placeholder for the methods defined below
        """
        return None

    def sample(self, seed=None):
        """
        Return a noisy power spectrum sampled from the probability distribution
        of the noisy spectra
        """
        return noisy_power_spectrum(self.power(), seed=seed)


class ConstantSpectrum(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.constant(self.frequencies, self.parameters)


class SimplePowerLawSpectrum(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.power_law(self.frequencies, self.parameters)


class SimplePowerLawSpectrumWithConstantBackground(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.power_law_with_constant(self.frequencies, self.parameters)


class BrokenPowerLaw(PowerLawPowerSpectrum):
    def power(self):
        return rnspectralmodels.broken_power_law_log_break_frequency(self.frequencies, self.parameters)


class TimeSeriesFromPowerSpectrum():
    def __init__(self, powerspectrum, V=10, W=10, seed=None, fft_zero=0.0, **kwargs):
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
        powerspectrum : PowerLawPowerSpectrum object
            defines the properties of the power spectrum
        V : scalar >= 1
            oversampling parameter mimicing increasing the time series duration
            from N*dt to V*N*dt
        W : scalar >= 1
            oversampling parameter mimicing increasing the cadence from dt to
            dt/W

        fft_zero : scalar number
            The value at the zero Fourier frequency.

        seed : scalar number
            A seed value for the random number generator
        """
        self.powerspectrum = copy.deepcopy(powerspectrum)
        self.nt = self.powerspectrum.nt
        self.fft_zero = fft_zero

        # Vaughan (2010)
        self.V = V
        self.W = W

        # Random number seed
        self.seed = seed

        #
        self.K = self.V * self.W * self.nt / 2
        self.dt = self.powerspectrum.dt

        # Frequencies that we are calculating the power spectrum at
        self.frequencies = np.arange(1, self.K + 1) / (self.V * self.nt * self.dt)
        self.powerspectrum.frequencies = self.frequencies
        self.inputpower = self.powerspectrum.power()

        # the fully over-sampled timeseries, with a sampling cadence of dt/W, with a
        # duration of V*N*dt
        self.oversampled = time_series_from_power_spectrum(self.inputpower, fft_zero=self.fft_zero, seed=self.seed, **kwargs)

        # Subsample the time-series back down to the requested cadence of dt
        self.longtimeseries = self.oversampled[0:len(self.oversampled):self.W]
        nlts = len(self.longtimeseries)

        # get a sample of the desired length nt from the middle of the long time series
        self.sample = self.longtimeseries[nlts / 2 - self.nt / 2: nlts / 2 - self.nt / 2 + self.nt]


def equally_spaced_nonzero_frequencies(n, dt):
    """
    Create a set of equally spaced Fourier frequencies
    """
    all_fft = np.fft.fftfreq(n, d=dt)
    return all_fft[all_fft > 0]


def noisy_power_spectrum(S, seed=None):
    """
    Create a noisy power spectrum, given some input power spectrum S, following
    the recipe of Vaughan (2010), MNRAS, 402, 307, appendix B

    Parameters
    ----------
    S : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.

    seed : scalar number
        A seed value for the random number generator
    """
    # Number of positive frequencies to calculate.
    K = len(S)

    # chi-squared(2) random numbers for all frequencies except the nyquist
    # frequency
    X = np.concatenate((chi2.rvs(2, size=K - 1, seed=seed),
                        chi2.rvs(1, size=1, seed=seed)))
    # power spectrum
    return S * X / 2.0


def time_series_from_power_spectrum(S, fft_zero=0.0, seed=None, no_noise=False):
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

    # noisy power spectrum
    if no_noise:
        I = S
    else:
        I = noisy_power_spectrum(S, seed=seed)

    # random phases, except for the nyquist frequency.
    ph = uniform.rvs(loc=-np.pi / 2.0, scale=np.pi, size=K, seed=seed)
    if no_noise:
        ph[:] = 0.0
    ph[-1] = 0.0

    # Amplitudes
    A = np.sqrt(I / 2.0)

    # WARNING - SPECTRAL POWER APPEARs TO BE OUT BY A FACTOR 2 WITHOUT THIS
    # MULTIPLICATION FACTOR BELOW
    A = A * np.sqrt(2.0)

    # Complex vector
    F = A * np.exp(-np.complex(0, 1) * ph)

    # Form the negative frequency part
    #F_negative = np.conjugate(F)[::-1][1:]

    # Form the fourier transform
    #F_complete = np.concatenate((np.asarray([fft_zero]), F, F_negative))

    # create the time-series.  The complex part should be tiny.
    #T_sim = np.fft.ifft(F_complete)
    T_sim = np.fft.irfft(np.concatenate((np.asarray([fft_zero]), F)))

    # The time series is formally complex.  Return the real part only.
    return np.real(T_sim)
