"""
Simple time series object
"""

import numpy as np
from matplotlib import pyplot as plt


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

        # Mean power
        self.vaughan_mean = np.mean(self.ppower)

        # Power spectrum normalized by its mean
        self.normed_by_mean = self.ppower / self.vaughan_mean

        # Standard deviation of the normalized power
        self.vaughan_std = np.std(self.normed_by_mean)

        # Normalized power expressed in units of its standard deviation
        self.Npower = self.normed_by_mean / self.vaughan_std

    def peek(self):
        """
        Generates a quick plot of the positive frequency part of the power
        spectrum.
        """
        plt.plot(self.frequencies.positive, self.ppower)


class TimeSeries:
    def __init__(self, time, data, label='data', units=None):
        self.SampleTimes = SampleTimes(time)
        if self.SampleTimes.nt != data.size:
            raise ValueError('length of sample times not the same as the data')
        self.data = data
        self.PowerSpectrum = PowerSpectrum(np.fft.fftfreq(self.SampleTimes.nt, self.SampleTimes.dt),
                                           np.abs(np.fft.fft(self.data)) ** 2)
        self.label = label
        self.units = units

    def peek(self):
        """
        Generates a quick plot of the data
        """
        plt.plot(self.SampleTimes.time, self.data)
        plt.xlabel(self.SampleTimes.label + " " + self.SampleTimes.units)
        if self.units is not None:
            plt.ylabel(self.label + self.units)
        else:
            plt.ylabel(self.label)
        plt.show()
