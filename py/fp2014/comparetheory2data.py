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
from general import theory_TS, imgdir, data_TS2, spectrum_exp_decay


# plot the frequencies in this unit
requested_unit = 'mHz'

# Subsample the data at this image cadence
subsampler = [12]

waveband = {}

for subsample in subsampler:
    theory_TS = theory_TS(12)
    data_TS = data_TS2()

    for i, kk in enumerate(theory_TS.keys()):
    # get the data and subsample it
        plt.figure(i)

        data = theory_TS[kk]["data"][0: 1800]
        t = theory_TS[kk]["t"][0: 1800]
        wb = theory_TS[kk]["wb"]
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

        data2 = data_TS[kk]["data"]
        data2 = data2 * np.std(data) / np.std(data2)
        t2 = data_TS[kk]["t"]
        wb2 = data_TS[kk]["wb"]
        nt2 = len(data2)

        # timeseries
        ts2 = TimeSeries(t2,
                        data2 * np.hanning(nt2),
                        base=datetime.datetime(2000, 1, 1))

        # Get the location of the positive frequencies
        posindex2 = ts2.PowerSpectrum.F.frequencies > 0

        # Get the data we wish to plot
        power2 = ts2.PowerSpectrum.P[posindex]
        pfreq2 = ts2.PowerSpectrum.F.frequencies[posindex].to(requested_unit)
        # Plot
        with plt.style.context(('ggplot')):
            plt.plot(t2, data2, label='Viall & Klimchuk 2012 observed data')
            plt.plot(t, data, label='Viall & Klimchuk 2013 Fig. 1, modeled light curves')
            #plt.loglog(pfreq, spectrum_exp_decay(pfreq.to('Hz').value, 2000.0))
            plt.xlabel('time (seconds) [%i samples]' % (nt2))
            plt.ylabel('Emission (arb. units)')
            plt.title(wb + ': comparison of time series')
            plt.tight_layout()
    #plt.axvline(1.0 / (6 * 60 * 60.0) * u.Hz.to(requested_unit), label='Ireland et al (2014) frequency range', color='k', linestyle= ':')
    #plt.axvline(1.0 / (2 * 12) * u.Hz.to(requested_unit), color='k', linestyle= ':')
    #plt.axvline(1.0 / (300.0) * u.Hz.to(requested_unit), label='5 minutes', color='k', linestyle= '-')
    #plt.axvline(1.0 / (180.0) * u.Hz.to(requested_unit), label='3 minutes', color='k', linestyle= '--')
        plt.legend(fontsize=12, loc=3)
        savefig = os.path.join(imgdir, 'compare_VK2013_TS_theory_modeled_diffuse_todata.%s.png' % (kk))
        plt.savefig(savefig)
        plt.close('all')

        # Plot
        displacement = 3.0
        with plt.style.context(('ggplot')):
            plt.loglog(pfreq2, power2, label='Viall & Klimchuk 2012 observed data')
            plt.loglog(pfreq, power / (10.0 ** displacement), label='Viall & Klimchuk 2013 Fig. 1, modeled light curves')
            #plt.loglog(pfreq, spectrum_exp_decay(pfreq.to('Hz').value, 2000.0))
            plt.xlabel('frequency (%s) [%i frequencies]' % (pfreq.unit, len(pfreq)))
            plt.ylabel('Fourier power (arb.units)')
            plt.title(wb + ': comparison of Fourier power spectra')
            plt.ylim(10.0 ** (-11 - displacement), 10.0 ** 1)
            plt.tight_layout()
    #plt.axvline(1.0 / (6 * 60 * 60.0) * u.Hz.to(requested_unit), label='Ireland et al (2014) frequency range', color='k', linestyle= ':')
    #plt.axvline(1.0 / (2 * 12) * u.Hz.to(requested_unit), color='k', linestyle= ':')
    #plt.axvline(1.0 / (300.0) * u.Hz.to(requested_unit), label='5 minutes', color='k', linestyle= '-')
    #plt.axvline(1.0 / (180.0) * u.Hz.to(requested_unit), label='3 minutes', color='k', linestyle= '--')
        plt.legend(fontsize=12, loc=3)
        savefig = os.path.join(imgdir, 'compare_VK2013_theory_modeled_diffuse_todata.%s.png' % (kk))
        plt.savefig(savefig)
        plt.close('all')
