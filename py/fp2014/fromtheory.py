#
# plot out some power spectra that arise from theory
#
import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
from timeseries import TimeSeries
import datetime
import astropy.units as u
import os
from general import norm_hanning, imgdir

location = '/home/ireland/ts/sav/NV/Jack_model_TS.sav'

nvdata = readsav(location)

# plot the frequencies in this unit
requested_unit = 'mHz'

# Subsample the data at this image cadence
subsampler = [12]

waveband = {}


for i, k in enumerate(nvdata.keys()):
    # get the data and subsample it

    plt.figure(i)
    for subsample in subsampler:

        data = nvdata[k][::subsample]

        # Normalize and apply the Hanning window
        data = norm_hanning(data)

        waveband[k] = k[1:] + '$\AA$'

        # length of the time series
        nt = len(data)

        # time
        base = datetime.datetime(2000, 1, 1)
        time = [base + datetime.timedelta(seconds=subsample * j) for j in xrange(nt)]

        # timeseries
        ts = TimeSeries(time, data)

        # Get the location of the positive frequencies
        posindex = ts.PowerSpectrum.F.frequencies > 0

        # Get the data we wish to plot
        power = ts.PowerSpectrum.P[posindex]
        pfreq = ts.PowerSpectrum.F.frequencies[posindex].to(requested_unit)

        # Plot
        with plt.style.context(('ggplot')):
            plt.loglog(pfreq, power, label='%i second cadence, %i samples' % (subsample, nt))
            plt.xlabel('frequency (%s) [%i frequencies]' % (pfreq.unit, len(pfreq)))
            plt.ylabel('Fourier power (arb.units)')
            plt.title('VK2013: ' + waveband[k] + ': modeled diffuse AR emission')
            plt.ylim(10 ** -13.0, 10.0 ** 0)
    #plt.axvline(1.0 / (6 * 60 * 60.0) * u.Hz.to(requested_unit), label='Ireland et al (2014) frequency range', color='k', linestyle= ':')
    #plt.axvline(1.0 / (2 * 12) * u.Hz.to(requested_unit), color='k', linestyle= ':')
    #plt.axvline(1.0 / (300.0) * u.Hz.to(requested_unit), label='5 minutes', color='k', linestyle= '-')
    #plt.axvline(1.0 / (180.0) * u.Hz.to(requested_unit), label='3 minutes', color='k', linestyle= '--')
    plt.legend(framealpha=0.5, fontsize=8)
    savefig = os.path.join(imgdir, 'VK2013_theory_modeled_diffuse_ar_emission.%s.png' % (k))
    plt.savefig(savefig)
plt.close('all')
