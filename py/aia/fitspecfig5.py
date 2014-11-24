#
#
#
import numpy as np

import aia_specific
import pymcmodels2
import rnspectralmodels
from paper1 import sunday_name, prettyprint, log_10_product, indexplot
from paper1 import csv_timeseries_write, pkl_write, fix_nonfinite, fit_details, get_kde_most_probable
from aia_pymc_pwrlaws_helpers import *
import matplotlib.pyplot as plt
from timeseries import TimeSeries
import fakedata

__author__ = 'ireland'
plt.ion()
# Reproducible

def do(t, data, testing=1, window=False):
    nt = t.size
    print 'Fitting using pymc'
    dir = os.path.expanduser('~/ts/fixfig5')

    if window == True:
        win = np.hanning(nt)
        winname = 'hanning'
        print('Applying the Hanning window.')
    else:
        win = 1.0
        winname = 'none'
        print('No windows applied.')

    meandata = np.mean(data)
    data = (data - meandata)/meandata
    itera = 50000 /testing
    burn = 10000 / testing

    ts = TimeSeries(t, data * win)

    pwr = ts.PowerSpectrum.ppower
    pfreqs = ts.PowerSpectrum.frequencies.positive
    pymcmodel0 = pymcmodels2.spl(pfreqs, pwr)
    M0 = pymc.MCMC(pymcmodel0)

    # Run the sampler
    M0.sample(iter=itera, burn=burn, thin=5, progress_bar=True)

    # Get the mean results and the 95% confidence level
    mean = M0.stats()['fourier_power_spectrum']['mean']
    low = M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]
    high = M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]
    index = M0.stats()['power_law_index']['mean']

    np.save(os.path.join(dir, 'testdata_best_fit_pwr.%s.npy' % (winname)), mean)
    np.save(os.path.join(dir, 'testdata_best_fit_freqs.%s.npy' % (winname)), pfreqs)

    plt.close('all')
    plt.loglog(pfreqs, pwr, label='test data, power')
    plt.loglog(pfreqs, mean, label = 'best fit index = %f' %(index))
    plt.loglog(pfreqs, low, label = '95 %')
    plt.loglog(pfreqs, high, label = '95 %')
    plt.legend()
    plt.savefig(os.path.join(dir, 'fitspecfig5_output.%s.png' % (winname)))
    return M0
