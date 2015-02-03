"""
Simple time series object
"""

import numpy as np
from astropy import units as u
from sunpy.util.unit_decorators import quantity_input


class SampleTimes:
    @quantity_input(t=u.s)
    def __init__(self, t):
        """A class holding time series sample times."""

        # ensure that the initial time is zero
        self.t = t - t[0]

        # Average cadence
        self.dt = self.t[-1] / (len(self.t) - 1)

        # Cadences
        self.cadences = self.t[1:] - self.t[0:-1]

        # Include base time for input series
        self.basetime = t[0]

    # Number of sample times
    def __len__(self):
        return len(self.t)

    # Normalized dimensionless times
    def normalized(self, norm=None):
        if norm is None:
            normalization = self.dt.value
        else:
            normalization = norm.value
        return self.t.value / normalization * u.dimensionless_unscaled

    def segment_indices(self, absolutetolerance=0.5):
        """
        Find segments in the data where segments of the sample times have
        cadences below the specified absolute tolerance.

        Returns
        -------
        indices : list
            A list of lists.  Each list is the start and end index of
            the sample times that indicates the start and end of a
            segment of the sample times where the cadences are below
            the specified absolute tolerance.
        """
        # raw cadences
        n = len(self.cadences)
        segments = []
        istart = 0
        iend = 1
        while iend <= n - 2:
            c0 = self.cadences[istart]
            c1 = self.cadences[iend]
            while (np.abs(c1 - c0) < absolutetolerance) and (iend <= n - 2):
                iend = iend + 1
                c0 = self.cadences[istart]
                c1 = self.cadences[iend]
            segments.append([istart, iend])
            istart = iend
        return segments


class Frequencies:
    @quantity_input(time=u.Hz)
    def __init__(self, f):
        self.f = f
        self.pindex = self.f > 0
        self.pf = self.f[self.pindex]

    # Number of frequencies
    def __len__(self):
        return len(self.f)

    # Normalized dimensionless frequencies
    def normalized(self, norm=None):
        if norm is None:
            normalization = np.min(self.f[self.pindex].value)
        else:
            normalization = norm.value
        return self.f.value / normalization * u.dimensionless_unscaled


# TODO - define a proper FFT class.
class PowerSpectrum:
    @quantity_input(frequencies=u.Hz)
    def __init__(self, frequencies, power):
        self.f = Frequencies(frequencies)
        self.power = power
        self.ppower = self.power[self.f.pindex]


class TimeSeries:
    def __init__(self, time, data):
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
        if isinstance(time, SampleTimes):
            self.SampleTimes = time
        else:
            self.SampleTimes = SampleTimes(time)
        self.data = data
        if len(self.SampleTimes) != len(self.data):
            raise ValueError('Length of sample times not the same as the data')

        # Note that the power spectrum is defined
        self.fft_transform = np.fft.fft(self.data)
        nt = len(self.SampleTimes)
        self.PowerSpectrum = PowerSpectrum(np.fft.fftfreq(nt, self.SampleTimes.dt) * (1.0 / self.SampleTimes.t.unit),
                                           (np.abs(self.fft_transform) ** 2) / (1.0 * nt))

   # length of the time-series
    def __len__(self):
        return len(self.data)
