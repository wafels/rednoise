#
# plot out some power spectra that arise from theory
#
import os
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
from timeseries import TimeSeries
import datetime
import astropy.units as u
from general import theory_TS, imgdir, data_TS2, spectrum_exp_decay
#from paper1 import log_10_product

# plot the frequencies in this unit
requested_unit = 'mHz'

# Subsample the data at this image cadence
subsampler = [12]

waveband = {}

for subsample in subsampler:
    theory_TS = theory_TS(12)
    data_TS = data_TS2()

    for i, kk in enumerate(theory_TS.keys()):

        # Get the theoretical data
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

        #if kk == '193':
        #    dfjsd = df

        # Plot
        displacement = 0.0
        # Set the scale type on each axis
        #ax = plt.subplot(111)
        #ax.set_xscale('log')
        #ax.set_yscale('log')

        # Set the formatting of the tick labels
        #xformatter = plt.FuncFormatter(log_10_product)
        #ax.xaxis.set_major_formatter(xformatter)
        plt.loglog(pfreq2, power2, label='Viall & Klimchuk 2012 observed data')
        plt.loglog(pfreq, power / (10.0 ** displacement), label='Viall & Klimchuk 2013 Fig. 1, modeled light curves')
        #plt.loglog(pfreq, spectrum_exp_decay(pfreq.to('Hz').value, 2000.0))
        plt.xlabel('frequency (%s)' % (pfreq.unit))
        plt.ylabel('Fourier power (arb.units)')
        plt.title(wb + ': Fourier power spectra')
        plt.ylim(10.0 ** (-11 - displacement), 10.0 ** 1)
        plt.axhline(10.0 ** -5, color='k', linestyle="--")
        floc = 10.0 ** -1.9
        plt.text(floc, 10.0 ** -5.5, 'lower limit of Fourier power', color='k', fontsize=11, fontstyle='italic')
        plt.text(floc, 10.0 ** -6.0, 'in Figure 1 plots', color='k', fontsize=11, fontstyle='italic')
        plt.tight_layout()
        plt.legend(fontsize=12, loc=3, framealpha=0.5)
        savefig = os.path.join(imgdir, 'compare_VK2013_TS_theory_modeled_diffuse_todata.%s.png' % (kk))
        plt.savefig(savefig)
        plt.close('all')


        # Plot
        with plt.style.context(('ggplot')):
            plt.figure()
            plt.loglog(pfreq2, power2, label='Viall & Klimchuk 2012 observed data')
            plt.loglog(pfreq, power / (10.0 ** displacement), label='Viall & Klimchuk 2013 Fig. 1, modeled light curves')
            #plt.loglog(pfreq, spectrum_exp_decay(pfreq.to('Hz').value, 2000.0))
            plt.xlabel('frequency (%s)' % (pfreq.unit))
            plt.ylabel('Fourier power (arb.units)')
            plt.title(wb + ': Fourier power spectra')
            plt.ylim(10.0 ** (-11 - displacement), 10.0 ** 1)
            plt.tight_layout()
            plt.legend(fontsize=12, loc=3, framealpha=1.0)
            plt.axhline(10.0 ** -5)
            #plt.text(pfreq[0].to('mHz').value, 10.0 ** -5, 'lower limit of Figure 1 plots')
            savefig = os.path.join(imgdir, 'compare_VK2013_theory_modeled_diffuse_todata.ggplot.%s.png' % (kk))
            plt.savefig(savefig)
            plt.close('all')

