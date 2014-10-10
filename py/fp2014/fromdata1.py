#
# plot out some power spectra that arise from theory
#
import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
from timeseries import TimeSeries
import datetime

location = '/home/ireland/ts/sav/NV/Jack_data_TS.sav'

nvdata = readsav(location)

# plot the frequencies in this unit
requested_unit = 'mHz'

# Subsample the data at this image cadence
subsample = 1

waveband = {}


for i, k in enumerate(nvdata.keys()):
    # get the data and subsample it
    indata = nvdata[k][::subsample]

    base = datetime.datetime(2000, 1, 1)
    time = [base + datetime.timedelta(seconds=j) for j in np.float64(indata[1, :])]
    data = indata[1, :]

    waveband[k] = k[1:] + '$\AA$'

    # length of the time series
    nt = len(data)

    # Multiply by the Hanning window
    data = data * np.hanning(nt)

    # timeseries
    ts = TimeSeries(time, data)

    # Get the location of the positive frequencies
    posindex = ts.PowerSpectrum.F.frequencies > 0

    # Get the data we wish to plot
    power = ts.PowerSpectrum.P[posindex]
    pfreq = ts.PowerSpectrum.F.frequencies[posindex].to(requested_unit)

    # Plot
    with plt.style.context(('ggplot')):
        plt.figure(i)
        plt.loglog(pfreq, power)
        plt.xlabel('frequency (%s), %i frequencies, %i samples' % (pfreq.unit, len(pfreq), nt))
        plt.ylabel('Fourier power (arb.units)')
        plt.title(waveband[k] + ': data')
        #savefig = 'modeled_diffuse_ar_emission.%s.%i.png' % (k, subsample)
        #plt.savefig(savefig)
