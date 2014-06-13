#
# Code to fit power laws to some data from the region analysis
#
# Also do posterior predictive checking to find which model fits best.
#
import pickle
import numpy as np
import os
#from matplotlib import rc_file
#matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
#rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

import pymc
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# Normality tests
from scipy.stats import shapiro, anderson
from statsmodels.graphics.gofplots import qqplot

import aia_specific
import pymcmodels3
import rnspectralmodels
from paper1 import sunday_name, prettyprint, log_10_product
from paper1 import csv_timeseries_write, pkl_write, fix_nonfinite, fit_details

# Reproducible
np.random.seed(1)


#
# Functions to calculate and store results
#
def fit_summary(M, maxlogp, chisquared, k=None):
    # Calculate the log likelihood
    lnL = np.sum([x.logp for x in M.observed_stochastics])
    print 'fit_summary: lnL as calculated by PyMC ', lnL

    # Number of variables
    if k == None:
        k = M.len - M.data_len
    print 'fit_summary: number of variables in fit = %i' % (k)

    # Calculate the AIC
    AIC = 2. * (k - lnL)

    # Calculate the BIC
    BIC = k * np.log(M.data_len) - 2.0 * lnL

    return {"AIC": AIC, "BIC": BIC, "maxlogp": maxlogp, "chi2": chisquared}


def T_LRT(maxlogp0, maxlogp1):
    if maxlogp0 == None or maxlogp1 == None:
        return None
    else:
        return -2 * (maxlogp0 - maxlogp1)


# Final storage dictionary
def FitSummary(M0, l0_data, chi0, M1, l1_data, chi1):
    fitsummary = {"t_lrt": T_LRT(l0_data, l1_data),
                  "M0": fit_summary(M0, l0_data, chi0),
                  "M1": fit_summary(M1, l1_data, chi1)}
    fitsummary["dAIC"] = fitsummary["M0"]["AIC"] - fitsummary["M1"]["AIC"]
    fitsummary["dBIC"] = fitsummary["M0"]["BIC"] - fitsummary["M1"]["BIC"]
    return fitsummary


#
# Calculate a log likelihood
#
def get_log_likelihood(y, f, sigma):
    A = -0.5 * ((y - f) / sigma) ** 2
    B = -0.5 * np.log(2 * np.pi * sigma ** 2)
    return np.sum(B + A)


#
# Hypothesis 0 functions
#
def get_variables_M0(m):
    A = [m.value.power_law_norm, m.value.power_law_index, m.value.background]
    return A


def curve_fit_M0(x, y, p0, sigma):
    answer = curve_fit(rnspectralmodels.Log_splwc_CF, x, y, p0=p0, sigma=sigma)
    return answer[0]


def get_spectrum_M0(x, A):
    return rnspectralmodels.Log_splwc(x, A)


def get_likelihood_M0(map_M0, x, pwr, sigma, obstype):
    tau = 1.0 / (sigma ** 2)
    A0 = get_variables_M0(map_M0)[0:3]
    A0 = curve_fit_M0(x, pwr, A0, sigma)
    if obstype == '.logiobs':
        return pymc.normal_like(pwr, get_spectrum_M0(x, A0), tau)
    else:
        return pymc.lognormal_like(pwr, get_spectrum_M0(x, A0), tau)


def get_chi_M0(map_M0, x, pwr, sigma):
    A0 = get_variables_M0(map_M0)
    spectrum = get_spectrum_M0(x, A0)
    sse = np.sum(((pwr - spectrum) / sigma) ** 2)
    #print sse, np.size(pwr), len(A0)
    return sse / (1.0 * (np.size(pwr) - len(A0)))


def T_SSE(data, fit, sigma):
    sse = np.sum(((data - fit) / sigma) ** 2)
    return sse


def calculate_reduced_chi2(x, pwr, spectrum, sigma, k):
    sse = T_SSE(pwr, spectrum, sigma)
    return sse / (1.0 * (np.size(pwr) - k))


def get_ci(data, ci=0.68):
    """Calculate the credible interval by excluding an equal amount of
    probability at each end"""
    extreme_prob = 0.5 * (1.0 - ci)
    sz = data.size
    ordered = np.sort(data)
    return ordered[np.rint(extreme_prob * sz)], \
            ordered[np.rint((1.0 - extreme_prob) * sz)]


def write_plots(M, region_id, obstype, savefig, model, passnumber,
                bins=100, savetype='.png', extra_factors=None):
    if model == 'M0':
        variables = ['power_law_norm', 'power_law_index', 'background']
    if model == 'M1':
        variables = ['power_law_norm', 'power_law_index', 'background',
                 'gaussian_amplitude', 'gaussian_position', 'gaussian_width']
    if model == 'M2':
        variables = ['power_law_norm', 'power_law_index', 'background',
                 'gaussian_amplitude', 'gaussian_position', 'gaussian_width']

    s = M.stats()
    fontsize = 10
    alpha = 0.75
    facecolor = 'white'
    bbox = dict(facecolor=facecolor, alpha=alpha)

    # Storage to track the 2-d plots
    vpairs = []

    # Go through all the variables
    for v in variables:
        # Fix the data, and get better labels
        data, mean, v95_lo, v95_hi, v68_lo, v68_hi, vlabel = fix_variables(v,
                                                           M.trace(v)[:],
                                                           s[v],
                                                           extra_factors=extra_factors)

        # Save file name
        filename = '%s_%s_%s_%s__%s__%s_%s' % (savefig, model, str(passnumber), obstype, '1d', v, savetype)

        # Find the histogram
        h, _ = np.histogram(data, bins)

        # Y position for labels
        ypos = np.max(h)

        # Plot the histogram and label it
        plt.figure(1)
        plt.hist(data, bins=bins)
        plt.xlabel(v + vlabel)
        plt.ylabel('p.d.f.')
        plt.title(region_id + ' ' + obstype + ' %s' % (model))

        # Summary statistics of the measurement
        plt.axvline(mean, color='k', label='mean', linewidth=2)
        plt.axvline(v95_lo, color='k', linestyle='--', linewidth=2)
        plt.axvline(v95_hi, color='k', linestyle='--', linewidth=2, label='95% CI')
        plt.axvline(v68_lo, color='k', linestyle='-.', linewidth=2)
        plt.axvline(v68_hi, color='k', linestyle='-.', linewidth=2, label='68% CI')
        print '------------------------'
        print v, '68%', v68_lo, v68_hi
        print v, '95%', v95_lo, v95_hi
        print '------------------------'

        plt.text(mean, 0.90 * ypos, ' %f' % (mean), fontsize=fontsize, bbox=bbox)
        plt.text(v95_lo, 0.25 * ypos, ' %f' % (v95_lo), fontsize=fontsize, bbox=bbox)
        plt.text(v95_hi, 0.75 * ypos, ' %f' % (v95_hi), fontsize=fontsize, bbox=bbox)
        plt.text(v68_lo, 0.42 * ypos, ' %f' % (v68_lo), fontsize=fontsize, bbox=bbox)
        plt.text(v68_hi, 0.59 * ypos, ' %f' % (v68_hi), fontsize=fontsize, bbox=bbox)

        # Define the legend
        plt.legend(framealpha=alpha, fontsize=fontsize, loc=3)
        plt.savefig(filename)
        plt.close('all')

        #
        # Do the 2-d variables
        #
        for v2 in variables:
            if (v2 != v) and not(v2 + v in vpairs):

                # Update the list of variable pairs
                vpairs.append(v2 + v)
                vpairs.append(v + v2)

                # Fix the data, and get better labels
                data2, mean2, v95_lo2, v95_hi2, v68_lo2, v68_hi2, vlabel2 = fix_variables(v2,
                                                               M.trace(v2)[:],
                                                               s[v2],
                                                               extra_factors=extra_factors)
                # Scatter plot : save file name
                filename = '%s_%s_%s_%s__%s__%s_%s_%s' % (savefig, model, str(passnumber), obstype, '2dscatter', v, v2, savetype)

                plt.figure(1)
                plt.scatter(data, data2, alpha=0.02)
                plt.xlabel(v + vlabel)
                plt.ylabel(v2 + vlabel2)
                plt.title(region_id + ' ' + obstype + ' %s' % (model))
                plt.plot([mean], [mean2], 'go')
                plt.text(mean, mean2, '%f, %f' % (mean, mean2), bbox=dict(facecolor='white', alpha=0.5))
                plt.savefig(filename)
                plt.close('all')

                # 2-D histogram
                filename = '%s_%s_%s_%s__%s__%s_%s_%s' % (savefig, model, str(passnumber), obstype, '2dhistogram', v, v2, savetype)

                plt.figure(1)
                plt.hist2d(data, data2, bins=bins, norm=LogNorm())
                plt.colorbar()
                plt.xlabel(v + vlabel)
                plt.ylabel(v2 + vlabel2)
                plt.plot([mean], [mean2], 'wo')
                plt.text(mean, mean2, '%f, %f' % (mean, mean2), bbox=dict(facecolor='white', alpha=0.5))
                plt.title(region_id + ' ' + obstype + ' %s' % (model))
                plt.savefig(filename)
                plt.close('all')


#
# Hypothesis 1 functions
#
def get_variables_M1(m):
    A = [m.value.power_law_norm, m.value.power_law_index, m.value.background,
         m.value.gaussian_amplitude, m.value.gaussian_position,
         m.value.gaussian_width]
    return A


def curve_fit_M1(x, y, p0, sigma, bump_type, n_attempt_limit=10):
    p_in = p0
    n_attempt = 0
    success = False
    while (not success) and (n_attempt <= n_attempt_limit):
        try:
            if bump_type == 'lognormal':
                answer = curve_fit(rnspectralmodels.Log_splwc_AddLognormalBump2_CF, x, y, p0=p_in, sigma=sigma)
            if bump_type == 'normal':
                answer = curve_fit(rnspectralmodels.Log_splwc_AddNormalBump2_allexp_CF, x, y, p0=p_in, sigma=sigma)
            success = True
        except RuntimeError:
            print 'Model M1 curve fit did not work - try varying parameters a tiny bit'
            p_in = p0 + 0.1 * np.random.random(size=(6))
            n_attempt = n_attempt + 1
            print 'Attempt number: %i' % (n_attempt)
    if success:
        return answer[0]
    else:
        return p0


def get_spectrum_M1(x, A, bump_shape):
    if bump_shape == 'lognormal':
        return rnspectralmodels.Log_splwc_AddLognormalBump2(x, A)
    if bump_shape == 'normal':
        return rnspectralmodels.Log_splwc_AddNormalBump2_allexp(x, A)


def get_likelihood_M1(map_M1, x, pwr, sigma, bump_shape):
    tau = 1.0 / (sigma ** 2)
    A1 = get_variables_M1(map_M1)[0:6]
    A1 = curve_fit_M1(x, pwr, A1, sigma, bump_shape)
    return pymc.normal_like(pwr, get_spectrum_M1(x, A1, bump_shape), tau)


def get_chi_M1(map_M1, x, pwr, sigma, bump_shape):
    A1 = get_variables_M1(map_M1)
    spectrum = get_spectrum_M1(x, A1, bump_shape)
    sse = np.sum(((pwr - spectrum) / sigma) ** 2)
    return sse / (1.0 * (np.size(pwr) - len(A1)))


#
# The variables as used in the fitting algorithm are not exactly those used
# when the equation is written down.  We transform them back to variables
# that make sense when plotting
#
def fix_variables(variable, data, s, extra_factors):
    def f1(d, norm=1.0):
        return np.log10(norm * np.exp(d))

    def f2(d):
        return (d ** 2) / np.log(10.0)

    def f3(d):
        return np.exp(d)

    mean = s['mean']
    v95_lo = s['95% HPD interval'][0]
    v95_hi = s['95% HPD interval'][1]

    v68_lo, v68_hi = get_ci(data, ci=0.68)

    if variable == 'power_law_norm' or variable == 'background':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), f1(v68_lo), f1(v68_hi), ' $[\log_{10}(power)]$'

    if variable == 'gaussian_amplitude':
        return f3(data), f3(mean), f3(v95_lo), f3(v95_hi), f3(v68_lo), f3(v68_hi), ' $[power]$'

    if variable == 'gaussian_position':
        return f1(data, norm=extra_factors[0]),\
            f1(mean, norm=extra_factors[0]),\
            f1(v95_lo, norm=extra_factors[0]),\
            f1(v95_hi, norm=extra_factors[0]),\
            f1(v68_lo, norm=extra_factors[0]),\
            f1(v68_hi, norm=extra_factors[0]),\
            ' $[\log_{10}(frequency)]$'

    if variable == 'gaussian_width':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), f1(v68_lo), f1(v68_hi), ' [decades of frequency]'

    return data, mean, v95_lo, v95_hi, v68_lo, v68_hi, ''
