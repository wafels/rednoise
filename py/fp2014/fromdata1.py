#
# plot out some power spectra that arise from theory
#
import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
from timeseries import TimeSeries
import datetime
import os
from general import norm_hanning, imgdir, equally_spaced

location = '/home/ireland/ts/sav/NV/Jack_data_TS.sav'

indata = readsav(location)

# plot the frequencies in this unit
requested_unit = 'mHz'

waveband = {}


for i, k in enumerate(indata.keys()):
    # get the data and extract the samples and the sample times
    nvdata = indata[k][0, :]
    nvt = np.float64(indata[k][1, :])
    nvt = nvt - nvt[0]

    # number of data points
    nt = len(nvdata)

    # resample the data to the average cadence
    t = equally_spaced(nt, nvt[-1])
    data_original = np.interp(t, nvt, nvdata)

    # Normalize and multiply by the Hanning window
    data = norm_hanning(data_original)

    # timeseries
    ts = TimeSeries(t, data, base=datetime.datetime(2000, 1, 1))

    # Get the location of the positive frequencies
    posindex = ts.PowerSpectrum.F.frequencies > 0

    # Get the data we wish to plot
    power = ts.PowerSpectrum.P[posindex]
    pfreq = ts.PowerSpectrum.F.frequencies[posindex].to(requested_unit)

    # Plot
    waveband[k] = k[4:] + '$\AA$'
    """
    with plt.style.context(('ggplot')):
        plt.figure(i)
        plt.plot(ts.ObservationTimes.cadence)
        plt.xlabel('cadence number (%i cadences)' % (len(ts.ObservationTimes.cadence)))
        plt.ylabel('cadence (%s)' % (ts.ObservationTimes.cadence.unit))
        plt.title(waveband[k] + ': sample cadence, diffuse emission lightcurve')
        savefig = os.path.join(imgdir, 'cadence.%s.png' % (k))
        plt.savefig(savefig)
    """
    with plt.style.context(('ggplot')):
        plt.figure(i + 10)
        plt.subplot(211)
        plt.plot(t, data_original)
        plt.xlabel('time (seconds) [%i samples]' % (len(t)))
        plt.ylabel('normalized counts')
        plt.xlim(t[0], t[-1])
        plt.title('VK2012: ' + waveband[k] + ': data')
        plt.subplot(212)
        plt.loglog(pfreq, power, label='%i samples' % (nt))
        plt.xlabel('frequency (%s) [%i frequencies]' % (pfreq.unit, len(pfreq)))
        plt.ylabel('Fourier power (arb.units)')
        plt.title('VK2012: ' + waveband[k] + ': Fourier power spectrum')
        plt.tight_layout()
        savefig = os.path.join(imgdir, 'VK2012_data_diffuse_emission_power_spectrum.%s.png' % (k))
        plt.savefig(savefig)

plt.close('all')
