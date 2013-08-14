from __future__ import absolute_import
"""
Co-align a set of maps and make sure the
"""

# Test 4: Randomly subsample the spatial positions
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os


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

    return results, x1, x2, x3


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

def make_plot_stuff(hinput, xh):
    hreturn = hinput / np.sum(hinput)
    nxh = xh.size
    x = 0.5 * (xh[0:nxh - 1] + xh[1:nxh])
    h_mean = np.sum(hreturn * x)
    h_std = np.sqrt(np.sum(hreturn * (x - h_mean) ** 2))
    return x, hreturn, h_mean, h_std

xlabel = ['power law index', 'ln(normalization)', 'ln(normalized background)']
qlabel = ['a', 'B', 'C']

for pindex in range(0, 3):
    plt.figure(pindex)
    plt.xlabel(xlabel[pindex])
    plt.ylabel('fraction found')

    for k, f in enumerate(files):
        results, x0, x1, x2 = loadresults(f)
        nsample = results.shape[1]
        plt.title('Posterior PDF averaged over %i randomly selected pixels' % (nsample))
        xinput = [x0, x1, x2]
        hinput = np.sum(results, axis=1)[pindex]
        x, hreturn, h_mean, h_std = make_plot_stuff(hinput, xinput[pindex])
        plt.plot(x, hreturn, label=r"%s: $\overline{%s}= %4.2f \pm %4.2f$" % (filelabel[k], qlabel[pindex], h_mean, h_std))

    plt.legend()
    plt.show()
