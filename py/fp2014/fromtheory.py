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
from general import theory_TS, imgdir


# plot the frequencies in this unit
requested_unit = 'mHz'

# Subsample the data at this image cadence
subsampler = [12]

waveband = {}

for subsample in subsampler:
    output = theory_TS(12)

    for i, kk in enumerate(output.keys()):
    # get the data and subsample it
        plt.figure(i)

        data = output[kk]["data"]
        t = output[kk]["t"]
        wb = output[kk]["wb"]
        nt = len(data)

        # timeseries
        ts = TimeSeries(t,
                        data * np.hanning(nt),
                        base=datetime.datetime(2000, 1, 1))

        # Get the location of the positive frequencies
        posindex = ts.PowerSpectrum.F.frequencies > 0

        # Get the data we wish to plot
        power = ts.PowerSpectrum.P[posindex]
        pfreq = ts.PowerSpectrum.F.frequencies[posindex].to(requested_unit)

        # Plot
        with plt.style.context(('ggplot')):
            plt.subplot(211)
            plt.plot(t, data)
            plt.xlabel('time (seconds) [%i samples]' % (len(t)))
            plt.ylabel('normalized counts')
            plt.title('VK2013: ' + wb + ': modeled diffuse AR emission')
            plt.subplot(212)
            plt.loglog(pfreq, power, label='%i second cadence, %i samples' % (subsample, nt))
            plt.xlabel('frequency (%s) [%i frequencies]' % (pfreq.unit, len(pfreq)))
            plt.ylabel('Fourier power (arb.units)')
            plt.title('VK2013: ' + wb + ': Fourier power spectrum')
            plt.ylim(10 ** -13.0, 10.0 ** 0)
            plt.tight_layout()
    #plt.axvline(1.0 / (6 * 60 * 60.0) * u.Hz.to(requested_unit), label='Ireland et al (2014) frequency range', color='k', linestyle= ':')
    #plt.axvline(1.0 / (2 * 12) * u.Hz.to(requested_unit), color='k', linestyle= ':')
    #plt.axvline(1.0 / (300.0) * u.Hz.to(requested_unit), label='5 minutes', color='k', linestyle= '-')
    #plt.axvline(1.0 / (180.0) * u.Hz.to(requested_unit), label='3 minutes', color='k', linestyle= '--')
        plt.legend(framealpha=0.5, fontsize=8)
        savefig = os.path.join(imgdir, 'VK2013_theory_modeled_diffuse_ar_emission.%i.%s.png' % (subsample, kk))
        plt.savefig(savefig)
        plt.close('all')