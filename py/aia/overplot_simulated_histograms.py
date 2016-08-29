#
# Make a plot of the simulated power law index histograms
#
import os
import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import details_study as ds

from scipy.stats import anderson_ksamp
from skimage.measure import compare_ssim, compare_nrmse

map_title = {}
subtitle = {}
my_map = {}


def normalized_dot_product(a, b):
    """
    From http://stackoverflow.com/questions/25977/how-can-i-measure-the-similarity-between-two-images .

    """
    return np.sum(a*b)**2/(np.sum(a**2)*np.sum(b**2))

plt.close('all')
for simulation in ds.simulation:

    filename = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/{:s}/disk/sim0/171/six_euv/spatial.common.171.powerlawindex.six_euv.standard.pkl'.format(simulation)

    print('Loading {:s}'.format(filename))
    f = open(filename, 'rb')
    map_title[simulation] = pickle.load(f)
    subtitle[simulation] = pickle.load(f)
    my_map[simulation] = pickle.load(f)
    f.close()

    md = my_map[simulation].compressed()
    bins = 50
    weights = np.ones_like(md)/len(md)
    plt.hist(md, bins=bins, weights=weights, label=ds.sim_name[simulation], alpha=0.35)
    plt.xlabel('power law index (n)')
    plt.ylabel('fraction in bin')
    plt.title('comparison across simulations')

subsample = 10
sim1 = "papern_bradshaw_simulation_intermediate_fn"
sim2 = "papern_bradshaw_simulation_high_fn"
a1 = my_map[sim1].compressed()[0::subsample]
# a1 = a1[a1 > 0.2]
a2 = my_map[sim2].compressed()[0::subsample]
# a2 = a2[a2 > 0.2]

print(' ')
print('Anderson-Darling k-sample test between {:s} and {:s}'.format(sim1, sim2))
ad = anderson_ksamp([a1, a2])
sig_string = "intermediate vs. high\n\n"
sig_string += "Anderson-Darling k-sample test\n"
sig_string += "  significance level = %3.3f%%\n" % (100*ad.significance_level)
sig_string += "SSIM (global) = %3.2f" % compare_ssim(my_map[sim1], my_map[sim2])
print(ad)
plt.text(3.2, 0.12, sig_string, fontstyle='italic', bbox=dict(facecolor='yellow', alpha=0.1))

plt.legend()
filepath = os.path.join('/home/ireland/Desktop', 'power_index_comparison_across_simulations.png')
print('Saving to ' + filepath)
plt.savefig(filepath, bbox_inches='tight')
plt.close('all')

# SSIM
for simulation1 in ds.simulation:
    m1 = my_map[simulation1]
    for simulation2 in ds.simulation:
        m2 = my_map[simulation2]
        ssim12 = compare_ssim(m1, m2)
        print(' ')
        print('SSIM (masked) {:s} vs {:s} = {:n} '.format(simulation1, simulation2, ssim12))
        ssim12u = compare_ssim(m1.data, m2.data)
        print('SSIM (unmasked) {:s} vs {:s} = {:n} '.format(simulation1, simulation2, ssim12))
        m1z = deepcopy(m1.data)
        m1z[m1z < 1.0] = 0.0
        m2z = deepcopy(m2.data)
        m2z[m2z < 1.0] = 0.0
        ssim12uz = compare_ssim(m1z, m2z)
        print('SSIM (unmasked < 1 set to zero) {:s} vs {:s} = {:n} '.format(simulation1, simulation2, ssim12))
        print(normalized_dot_product(m1.data, m2.data))

n = 32
nx = my_map[sim1].shape[1]
ny = my_map[sim1].shape[0]
dy = ny // n
dx = nx // n
ssim = np.zeros((n, n))
for j in range(0, n):
    j1 = j * dy
    j2 = (j + 1) * dy - 1
    for i in range(0, n):
        i1 = i * dx
        i2 = (i + 1) * dx - 1
        smap1 = np.transpose(my_map[sim1].data)[j1:j2, i1:i2]
        smap2 = np.transpose(my_map[sim2].data)[j1:j2, i1:i2]
        ssim[j, i] = compare_ssim(smap1, smap2)

plt.imshow(ssim, interpolation='none', cmap=cm.viridis, origin='lower', extent=[0, ny-1, 0, nx-1], vmin=-1, vmax=1)
plt.xlabel('pixels')
plt.ylabel('pixels')
title = "local SSIM\n"
title += "intermediate vs. high\n"
title += "calculated on {:n}$\\times${:n} superpixels".format(dy, dx)
plt.title(title)
cb = plt.colorbar()
cb.set_label('SSIM')
filepath = os.path.join('/home/ireland/Desktop', 'SSIM_power_index_comparison_across_simulations.png')
print('Saving to ' + filepath)
plt.savefig(filepath, bbox_inches='tight')
plt.close('all')

