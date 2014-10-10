"""
Simple time series object
"""

import numpy as np
from matplotlib import pyplot as plt
import tsutils
from astropy import units as u
from sunpy.time import parse_time


class ObservationTimes:
    def __init__(self, obstimes):
        self.time = parse_time(obstimes)

        # ensure that the initial time is zero and that the units are seconds
        self.sampletimes = (self.time - self.time[0]).total_seconds() * u.s

        # Calculate the average spacing
        self.dt = (self.sampletimes[-1] - self.sampletimes[0]) / (1.0 * self.__len__)

    def __len__(self):
        return len(self.time)


class Frequencies:
    def __init__(self, frequencies):
        if isinstance(frequencies, u.Quantity):
            if frequencies.unit.physical_type == u.Hz.physical_type:
                self.frequencies = frequencies
            else:
                print('input array has the wrong units')
        else:
            # Assume the user input a numpy array which has units of hertz
            self.frequencies = frequencies * u.Hz


class PowerSpectrum:
    def __init__(self, frequencies, power):
        self.F = Frequencies(frequencies)
        self.P = power


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
        self.ObservationTimes = ObservationTimes(time)
        nt = len(self.ObservationTimes)
        if nt != data.shape[-1]:
            raise ValueError('Length of sample times not the same as the data.')

        # Assign the data
        self.data = data

        # Note that the power spectrum is defined
        self.fft_transform = np.fft.fft(self.data)
        self.PowerSpectrum = PowerSpectrum(np.fft.fftfreq(nt, self.ObservationTimes.dt),
                                           (np.abs(self.fft_transform) ** 2) / (1.0 * nt))

