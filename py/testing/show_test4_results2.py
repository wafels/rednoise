from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import rnspectralmodels


def loadresults(filename):
    print('Loading ' + filename)
    loadfile = open(filename, 'rb')
    data = pickle.load(loadfile)
    locations = pickle.load(loadfile)
    results = pickle.load(loadfile)
    x1 = pickle.load(loadfile)
    x2 = pickle.load(loadfile)
    x3 = pickle.load(loadfile)
    map_fit = pickle.load(loadfile)
    loadfile.close()

    return data, locations, results, x1, x2, x3, map_fit
 

def create_hist_plot(hist, bins, width=0.7, facecolor='blue'):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, facecolor=facecolor)

files = [os.path.expanduser('~/ts/pickle/test4_traditional_model/test4_traditional_model.all_samples.12.pickle')]
filelabel = ['simulated']
"""
files = [os.path.expanduser('~/ts/pickle/jul29_keep/SOL2011-04-30T21-45-49L061C108/171/test4.rand_locations.pickle'),
         os.path.expanduser('~/ts/pickle/jul29_keep/SOL2011-04-30T21-45-49L061C108/193/test4.rand_locations.pickle')]
filelabel = ['171', '193']
"""
"""
files = [os.path.expanduser('~/ts/pickle/jul30/SOL2011-05-09T22-28-13L001C055/171/test4.rand_locations.pickle'),
         os.path.expanduser('~/ts/pickle/jul30/SOL2011-05-09T22-28-13L001C055/193/test4.rand_locations.pickle')]
filelabel = ['171', '193']
"""

#f = os.path.expanduser('~/ts/pickle/jul30/SOL2011-05-09T22-28-13L001C055/193/test4.full_ts.pickle')
f = os.path.expanduser('~/ts/pickle/jul30/SOL2011-04-30T21-45-49L061C108/193/test4.full_ts.pickle')

data, locations, results, x1, x2, x3, map_fit = loadresults(f)

nt = data.shape[0]

freq = np.fft.fftfreq(nt, 12.0)
fpos = freq > 0
f = freq[fpos]

pwr = (np.abs(np.fft.fft(data)) ** 2)[fpos]

m = map_fit.flatten()

map_pwr = rnspectralmodels.power_law_with_constant(f, [m[1], m[0], m[2]])

mean = np.zeros(shape=3)
x1m = 0.5 * (x1[0:100] + x1[1:101])
x2m = 0.5 * (x2[0:100] + x2[1:101])
x3m = 0.5 * (x3[0:100] + x3[1:101])

mean[0] = np.sum(x1m * results[0, 0, :]) / np.sum(results[0, 0, :])
mean[1] = np.sum(x2m * results[1, 0, :]) / np.sum(results[1, 0, :])
mean[2] = np.sum(x3m * results[2, 0, :]) / np.sum(results[2, 0, :])

mean_pwr = rnspectralmodels.power_law_with_constant(f, [mean[1], mean[0], mean[2]])

plt.loglog(f, pwr/pwr[0])
plt.loglog(f, map_pwr, label='MAP')
plt.loglog(f, mean_pwr, label='mean')
plt.legend()
plt.ylabel('normalized power')
plt.xlabel('frequency (Hz)')
plt.show()