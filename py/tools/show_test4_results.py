from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import scipy 
import numpy as np
from matplotlib import pyplot as plt
import sunpy
import pickle
import os


def loadresults(filename):
    print('Loading '+ filename)
    loadfile = open(filename, 'rb')
    data = pickle.load(loadfile)
    locations = pickle.load(loadfile)
    results = pickle.load(loadfile)
    x1 = pickle.load(loadfile)
    x2 = pickle.load(loadfile)
    x3 = pickle.load(loadfile)
    loadfile.close()

    return results, x1, x2, x3


def create_hist_plot(hist, bins, width=0.7, facecolor='blue'):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, facecolor=facecolor)

f171 = os.path.expanduser('~/ts/pickle/jul29/SOL2011-04-30T21-45-49L061C108/171/test4.rand_locations.pickle')
r171, x1_171, x2_171, x3_171 = loadresults(f171)

f193 = os.path.expanduser('~/ts/pickle/jul29/SOL2011-04-30T21-45-49L061C108/193/test4.rand_locations.pickle')
r193, x1_193, x2_193, x3_193 = loadresults(f193)

nsample = r171.shape[1]

x = 0.5 * (x1_171[0:100] + x1_171[1:101])

h171 = np.sum(r171, axis=1)
h171[0, :] = h171[0, :] / np.sum(h171[0, :])
h171_mean = np.sum(h171[0, :] * x)
h171_std = np.sum(h171[0, :] * (x - h171_mean) ** 2)
#create_hist_plot(h171[0, :] / np.sum(h171[0, :]), x1_171)

h193 = np.sum(r193, axis=1)
h193[0, :] = h193[0, :] / np.sum(h193[0, :])
h193_mean = np.sum(h193[0, :] * x)
h193_std = np.sum(h193[0, :] * (x - h193_mean) ** 2)
#create_hist_plot(h193[0, :] / np.sum(h193[0, :]), x1_193, facecolor='green')

plt.plot(x, h171[0, :], label=r"171: $\overline{\alpha}= %4.2f \pm %4.2f$" % (h171_mean, h171_std))
plt.plot(x, h193[0, :], label=r'193: $\overline{\alpha}= %4.2f \pm %4.2f$' % (h193_mean, h193_std))
plt.legend()
plt.title('Posterior PDF averaged over %i randomly selected pixels' % (nsample))
plt.xlabel('power law index')
plt.ylabel('fraction found')
plt.show()
