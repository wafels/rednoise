"""
Simple time series object
"""

import numpy as np
from matplotlib import pyplot as plt
import tsutils


class SampleTimes:
    def __init__(self, time, label='time', units='seconds'):
        """A class holding time series sample times."""
        # ensure that the initial time is zero
        self.time = time - time[0]

        # Number of sample times
        self.nt = self.time.size

        # Average cadence
        self.dt = self.time[-1] / (self.nt - 1)

        # Information on the units for the time
        self.label = label
        self.units = units

        # Differences between consecutive sample times
        self.tdiff = self.time[1:] - self.time[0:-1]

        # Include base time for input series
        self.basetime=time[0]


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
        self.ppower = self.power[..., self.frequencies.posindex]
        self.label = label

        # Mean power
        self.vaughan_mean = np.mean(self.ppower)

        # Power spectrum normalized by its mean
        self.normed_by_mean = self.ppower / self.vaughan_mean

        # Standard deviation of the normalized power
        self.vaughan_std = np.std(self.normed_by_mean)

        # Normalized power expressed in units of its standard deviation
        self.Npower = self.normed_by_mean / self.vaughan_std

    def peek(self, **kwargs):
        """
        Generates a quick plot of the positive frequency part of the power
        spectrum.
        """
        plt.plot(self.frequencies.positive, self.ppower, **kwargs)


class TimeSeries:
    def __init__(self, time, data, label='data', units=None, name=None):
        """
        A simple object that defines a time-series object.  Handy for storing
        time-series and their (its) Fourier power spectra.  Fourier power
        spectra defined by this object are divided by the number of elements in
        the source time-series.  This is done so that the expected mathematical
        properties of the Fourier power spectrum are ensured.  For example,
        a time series of purely Gaussian-noisy data has the same Fourier power
        at all frequencies.  If the standard deviation of the Gaussian noise
        process is 1, then the Fourier power at all frequencies has an
        expectation value of 1.  Given the numpy definition of the FFT, to
        maintain this expectation value, this requires the Fourier power to be
        divided by the number of samples in the original time-series.
        """
        self.SampleTimes = SampleTimes(time)
        if self.SampleTimes.nt != data.shape[-1]:
            raise ValueError('length of sample times not the same as the data')
        self.nt = self.SampleTimes.nt
        self.data = data
        
        # Note that the power spectrum is defined
        self.PowerSpectrum = PowerSpectrum(np.fft.fftfreq(self.SampleTimes.nt, self.SampleTimes.dt),
                                           (np.abs(np.fft.fft(self.data)) ** 2) / (1.0 * self.nt))
        self.label = label
        self.units = units
        self.name = name
        self.pfreq = self.PowerSpectrum.frequencies.positive
        self.ppower = self.PowerSpectrum.ppower
        
        # Autocorrelation
        #self.acor = tsutils.autocorrelate(self.data)

    def peek(self, **kwargs):
        """
        Generates a quick plot of the data
        """
        plt.plot(self.SampleTimes.time, self.data, **kwargs)
        xunits = prepend_space(bracketize(self.SampleTimes.units))
        plt.xlabel(self.SampleTimes.label + xunits)
        nsamples = ' [%i samples]' % self.SampleTimes.nt
        if self.units is not None:
            yunits = prepend_space(bracketize(self.units))
            plt.ylabel(self.label + yunits + nsamples)
        else:
            plt.ylabel(self.label + nsamples)
        plt.title(self.name)

    def peek_ps(self, **kwargs):
        """
        Generates a quick plot of the power power spectrum of the data
        """
        plt.plot(self.PowerSpectrum.frequencies.positive,
                 self.PowerSpectrum.ppower, **kwargs)
        xunits = prepend_space(bracketize('%i frequencies' % len(self.pfreq)))
        plt.xlabel('Hz' + xunits)
        nsamples = ' [%i samples]' % self.SampleTimes.nt
        plt.ylabel('power' + nsamples)


def prepend_left_bracket(s, bracket='(', force_replace=False,
                          test_set=('(', '{', '[')):
    """Prepend a left bracket if possible"""
    if s[0] not in test_set:
        s = bracket + s
    else:
        if force_replace:
            s[0] = bracket
    return s


def append_right_bracket(s, bracket=')', force_replace=False,
                         test_set=(')', '}', ']')):
    """Append a left bracket if possible"""
    if s[-1] not in test_set:
        s = s + bracket
    else:
        if force_replace:
            s[-1] = bracket
    return s


def bracketize(s, bracket='()', force_replace=False):
    """Add enclosing brackets if need be"""
    s = prepend_left_bracket(s, bracket=bracket[0],
                             force_replace=force_replace)
    s = append_right_bracket(s, bracket=bracket[1],
                             force_replace=force_replace)
    return s


def prepend_space(s, force_replace=False):
    s = prepend_left_bracket(s, bracket=' ', force_replace=force_replace,
                              test_set=(' '))
    return s
