#
# Code to fit power laws to some data from the region analysis
#
# Also do posterior predictive checking to find which model fits best.
#
import os
import pickle
import numpy as np

import pymc
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import aia_specific
import pymcmodels
import rnspectralmodels

"""
#
# Curvefit the log of the power law plus constant with a Gaussian
#
def curve_fit_LogPowerLawPlusConstantGaussian(x, y, sigma, p0=None):
    A = curve_fit(aia_plaw.LogPowerLawPlusConstantGaussian, x, y, p0=p0)
    best_fit, likelihood = calc_LogPowerLawPlusConstantGaussian(x,
                                                   A[0][0],
                                                   A[0][1],
                                                   A[0][2],
                                                   A[0][3],
                                                   A[0][4],
                                                   A[0][5],
                                                   sigma)
    return A, best_fit, likelihood

def calc_LogPowerLawPlusConstantGaussian(x, A, sigma):
    best_fit = aia_plaw.LogPowerLawPlusConstantGaussian(x,
                                                   A[0],
                                                   A[1],
                                                   A[2],
                                                   A[3],
                                                   A[4],
                                                   A[5])
    likelihood = pymc.normal_like(best_fit, y, 1.0 / (sigma ** 2))

    return best_fit, likelihood


def calc_likelihood_gaussian(a, b, sigma):
    return pymc.normal_like(a, b, 1.0 / (sigma ** 2))


#
# Curvefit the log of the power law plus constant with a Gaussian
#
def curve_fit_LogPowerLawConstant(x, y, sigma, p0=None):
    A = curve_fit(aia_plaw.LogPowerLawPlusConstant, x, y, p0=p0)
    best_fit = aia_plaw.LogPowerLawPlusConstant(x,
                                                   A[0][0],
                                                   A[0][1],
                                                   A[0][2])
    likelihood = pymc.normal_like(best_fit, y, 1.0 / (sigma ** 2))

    return A, best_fit, likelihood
"""
np.random.seed(seed=1)


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


def get_likelihood_M0(map_M0, x, pwr, sigma, tau, obstype):
    A0 = get_variables_M0(map_M0)
    A0 = curve_fit_M0(x, pwr, A0, sigma)
    if obstype == '.logiobs':
        return pymc.normal_like(pwr, get_spectrum_M0(x, A0), tau)
    else:
        return pymc.lognormal_like(pwr, get_spectrum_M0(x, A0), tau)


def get_M0_plots(M, region_id, obstype, savefig, bins=100, savetype='.png'):
    variables = ['power_law_norm', 'power_law_index', 'background']
    s = M.stats()
    fontsize = 10
    alpha = 0.75
    facecolor = 'white'
    bbox = dict(facecolor=facecolor, alpha=alpha)
    for v in variables:
        h, _ = np.histogram(M.trace(v)[:], bins)
        ypos = 0.9 * np.max(h)
        plt.hist(M.trace(v)[:], bins=bins)
        plt.xlabel(v)
        plt.ylabel('p.d.f.')
        plt.title(region_id + ' ' + obstype + ' M0')

        # values
        plt.axvline(s[v]['mean'], color='k', label='mean', linewidth=2)
        plt.axvline(s[v]['95% HPD interval'][0], color='k', linestyle='-.', linewidth=2, label='95%, lower')
        plt.axvline(s[v]['95% HPD interval'][1], color='k', linestyle='--', linewidth=2, label='95%, upper')

        plt.text(s[v]['mean'], ypos, ' %f' % (s[v]['mean']), fontsize=fontsize, bbox=bbox)
        plt.text(s[v]['95% HPD interval'][0], ypos, ' %f' % (s[v]['95% HPD interval'][0]), fontsize=fontsize, bbox=bbox)
        plt.text(s[v]['95% HPD interval'][1], ypos, ' %f' % (s[v]['95% HPD interval'][1]), fontsize=fontsize, bbox=bbox)

        plt.legend(framealpha=0.5, fontsize=10)
        plt.savefig(savefig + obstype + '_M0_' + v + savetype)
        plt.close('all')


#
# Hypothesis 1 functions
#
def get_variables_M1(m):
    A = [m.value.power_law_norm, m.value.power_law_index, m.value.background,
         m.value.gaussian_amplitude, m.value.gaussian_position, m.value.gaussian_width]
    return A


def curve_fit_M1(x, y, p0, sigma, n_attempt_limit=10):
    p_in = p0
    n_attempt = 0
    success = False
    while (not success) and (n_attempt <= n_attempt_limit):
        try:
            answer = curve_fit(rnspectralmodels.Log_splwc_GaussianBump_CF, x, y, p0=p_in, sigma=sigma)
            success = True
        except RuntimeError:
            print 'Model M1 curve fit did not work - try varying parameters a tiny bit'
            p_in = p0 + 0.0001 * np.random.random(size=(6))
            n_attempt = n_attempt + 1
            print 'Attempt number: %i' % (n_attempt)
    return answer[0]


def get_spectrum_M1(x, A):
    return rnspectralmodels.Log_splwc_GaussianBump(x, A)


def get_likelihood_M1(map_M1, x, pwr, sigma, tau, obstype):
    A1 = get_variables_M1(map_M1)
    A1 = curve_fit_M1(x, pwr, A1, sigma)
    if obstype == '.logiobs':
        return pymc.normal_like(pwr, get_spectrum_M1(x, A1), tau)
    else:
        return pymc.lognormal_like(pwr, get_spectrum_M1(x, A1), tau)


def get_M1_plots(M, region_id, obstype, savefig, bins=100, savetype='.png'):
    variables = ['power_law_norm', 'power_law_index', 'background',
                 'gaussian_amplitude', 'gaussian_position', 'gaussian_width']

    s = M.stats()
    for v in variables:
        # Fix the data, and get better labels
        data, xlabel = fix_variables(v, M.trace(v)[:])

        # Find the histogram
        h, _ = np.histogram(data, bins)

        # Y position for labels
        ypos = 0.9 * np.max(h)

        # Plot the histogram and label it
        plt.hist(data, bins=bins)
        plt.xlabel(v + xlabel)
        plt.ylabel('p.d.f.')
        plt.title(region_id + ' ' + obstype + ' M0')

        # Summary statistics of the measurement
        plt.axvline(s[v]['mean'], color='k', label='mean', linewidth=2)
        plt.axvline(s[v]['95% HPD interval'][0], color='k', linestyle='-.', linewidth=2, label='95%, lower')
        plt.axvline(s[v]['95% HPD interval'][1], color='k', linestyle='--', linewidth=2, label='95%, upper')

        plt.text(s[v]['mean'], ypos, ' %f' % (s[v]['mean']), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
        plt.text(s[v]['95% HPD interval'][0], ypos, ' %f' % (s[v]['95% HPD interval'][0]), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
        plt.text(s[v]['95% HPD interval'][1], ypos, ' %f' % (s[v]['95% HPD interval'][1]), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))

        # Define the legend
        plt.legend(framealpha=0.5, fontsize=10, loc=3)
        plt.savefig(savefig + obstype + '_M1_' + v + savetype)
        plt.close('all')


def do_summary_histogram(variables, region_id, obstype, savefig, M, bins=100, savetype='.png'):
    s = M.stats()
    for v in variables:
        # Fix the data, and get better labels
        data, mean, v95_up, v95_lo, vlabel = fix_variables(v, M.trace(v)[:], s)

        # Find the histogram
        h, _ = np.histogram(data, bins)

        # Y position for labels
        ypos = 0.9 * np.max(h)

        # Plot the histogram and label it
        plt.hist(data, bins=bins)
        plt.xlabel(v + vlabel)
        plt.ylabel('p.d.f.')
        plt.title(region_id + ' ' + obstype + ' M0')

        # Summary statistics of the measurement
        plt.axvline(mean, color='k', label='mean', linewidth=2)
        plt.axvline(v95_lo, color='k', linestyle='-.', linewidth=2, label='95%, lower')
        plt.axvline(v95_up, color='k', linestyle='--', linewidth=2, label='95%, upper')

        plt.text(mean, ypos, ' %f' % (mean), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
        plt.text(v95_lo, ypos, ' %f' % (v95_lo), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
        plt.text(v95_up, ypos, ' %f' % (v95_up), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))

        # Define the legend
        plt.legend(framealpha=0.5, fontsize=10, loc=3)
        plt.savefig(savefig + obstype + '_M1_' + v + savetype)
        plt.close('all')


#
# The variables as used in the fitting algorithm are not exactly those used
# when the equation is written down.  We transform them back to variables
# that make sense when plotting
#
def fix_variables(variable, data, s):
    def f1(d):
        return d / np.log(10.0)

    def f2(d):
        return (d ** 2) / np.log(10.0)

    mean = s['mean']
    v95_lo = s['95% HPD interval'][0]
    v95_hi = s['95% HPD interval'][1]

    if variable == 'power_law_norm' or variable == 'background':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), '$[\log_{10}(power)]$'

    if variable == 'gaussian_amplitude':
        return f2(data), f2(mean), f2(v95_lo), f2(v95_hi), ' $[\log_{10}(power)]$'

    if variable == 'gaussian_position':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), ' $[\log_{10}(frequency)]$'

    if variable == 'gaussian_width':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), ' $[\log_{10}(frequency)]$'

    return data, mean, v95_lo, v95_hi, ''


plt.ioff()
plt.close('all')
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc/'
sfigroot = '~/ts/img_cc/'
scsvroot = '~/ts/csv_cc/'
corename = 'shutdownfun3_6hr'
sunlocation = 'disk'
fits_level = '1.5'
waves = ['171', '193']
regions = ['loopfootpoints', 'moss', 'sunspot', 'qs']
windows = ['hanning']
manip = 'relative'

nsample = 100

itera = 50000
burn = 1000
thin = 5

for iwave, wave in enumerate(waves):
    # Which wavelength?
    print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))

    # Now that the loading and saving locations are seot up, proceed with
    # the analysis.
    for iregion, region in enumerate(regions):
        # Which region
        print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))

        # Create the branches in order
        branches = [corename, sunlocation, fits_level, wave, region]

        # Set up the roots we are interested in
        roots = {"pickle": ldirroot,
                 "image": sfigroot,
                 "csv": scsvroot}

        # Data and save locations are based here
        locations = aia_specific.save_location_calculator(roots,
                                     branches)

        # set the saving locations
        sfig = locations["image"]
        scsv = locations["csv"]

        # Identifier
        ident = aia_specific.ident_creator(branches)

        # Go through all the windows
        for iwindow, window in enumerate(windows):
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Update the region identifier
            region_id = '.'.join((ident, window, manip))

            # Create a location to save the figures
            savefig = os.path.join(os.path.expanduser(sfig), window, manip)
            if not(os.path.isdir(savefig)):
                os.makedirs(savefig)
            savefig = os.path.join(savefig, region_id)

            for obstype in ['.iobs', '.logiobs']:
                #
                print('Analyzing ' + obstype)
                if obstype == '.iobs':
                    print('Data is lognormally distributed.')
                else:
                    print('Data is normally distributed.')

                # Load the data
                pkl_location = locations['pickle']
                ifilename = 'OUT.' + region_id + obstype
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                pwr_ff = pickle.load(pkl_file)
                pkl_file.close()

                # Normalize the frequency
                xnorm = freqs[0]
                x = freqs / xnorm

                # Set the Gaussian width
                sigma = 0.5 * np.log(10.0)
                tau = 1.0 / (sigma ** 2)

                #
                # Model 0 - the power law
                #
                # Geometric mean
                if obstype == '.logiobs':
                    pwr = pwr_ff
                    pymcmodel0 = pymcmodels.Log_splwc(x, pwr, sigma)
                # Arithmetic mean
                if obstype == '.iobs':
                    pwr = np.exp(pwr_ff)
                    pymcmodel0 = pymcmodels.Log_splwc_lognormal(x, pwr, sigma)

                # Initiate the sampler
                M0 = pymc.MCMC(pymcmodel0)

                # Run the sampler
                print('M0: Running simple power law model')
                M0.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)

                # Now run the MAP
                map_M0 = pymc.MAP(pymcmodel0)
                map_M0.fit(method='fmin_powell')

                # Plot the traces
                get_M0_plots(M0, region_id, obstype, savefig, bins=40)

                # Get the likelihood at the maximum
                #likelihood0_data = map_M0.logp_at_max
                l0_data = get_likelihood_M0(map_M0, x, pwr, sigma, tau, obstype)

                #
                # Second model - power law with Gaussian bumps
                #
                # Geometric mean
                if obstype == '.logiobs':
                    pwr = pwr_ff
                    pymcmodel1 = pymcmodels.Log_splwc_GaussianBump(x, pwr, sigma)
                # Arithmetic mean
                if obstype == '.iobs':
                    pwr = np.exp(pwr_ff)
                    pymcmodel1 = pymcmodels.Log_splwc_GaussianBump_lognormal(x, pwr, sigma)

                # Initiate the sampler
                M1 = pymc.MCMC(pymcmodel1)

                # Run the sampler
                print ' '
                print('M1: Running power law plus Gaussian model')
                M1.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)

                # Now run the MAP
                map_M1 = pymc.MAP(pymcmodel1)
                map_M1.fit(method='fmin_powell')

                # Plot the traces
                get_M1_plots(M1, region_id, obstype, savefig, bins=40)

                # Get the likelihood at the maximum
                likelihood1_data = map_M1.logp_at_max
                A1 = get_variables_M1(map_M1)
                A1[3] = 0.0
                A1 = curve_fit_M1(x, pwr, A1, sigma)
                l1_data = get_likelihood_M1(map_M1, x, pwr, sigma, tau, obstype)
                print l1_data

                # Get the T_lrt statistic for the data
                t_lrt_data = -2 * (l0_data - l1_data)
                print ' '
                print('M0 likelihood = %f, M1 likelihood = %f' % (l0_data, l1_data))
                print 'T_lrt for the initial data = %f' % (t_lrt_data)

                # Plot
                plt.figure(1)
                plt.loglog(freqs, np.exp(pwr_ff), label='data', color='k')

                # Plot the power law fit
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['mean']), label='power law, mean', color = 'b')
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='power law, 95% low', color = 'b', linestyle='-.')
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='power law, 95% high', color = 'b', linestyle='--')

                # Plot the power law and bump fit
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['mean']), label='power law + Gaussian, mean', color = 'r')
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='power law + Gaussian, 95% low', color = 'r', linestyle='-.')
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='power law + Gaussian, 95% high', color = 'r', linestyle='--')
                plt.legend(loc=3, framealpha=0.5, fontsize=10)
                plt.xlabel('frequency (Hz)')
                plt.ylabel('power')
                plt.title(obstype)
                plt.savefig(savefig + obstype + '.model_fit_compare.pymc.png')
                plt.close('all')
                """
                #
                # Do the posterior predictive comparison
                #
                t_lrt_distribution = []
                M0_logp = []
                M1_logp = []
                for r in range(0, nsample):
                    #
                    rthis = np.random.randint(0, high=8000)
                    print ' '
                    print r, nsample, rthis, ' (Positive numbers favor hypothesis 1 over 0)'
                    if obstype == '.logiobs':
                        # Generate test data under hypothesis 0
                        pred = M0.trace('predictive')[rthis]

                        # Fit the test data using hypothesis 0 and calculate a likelihood

                        M0_pp = pymcmodels.Log_splwc(x, pred, sigma)
                        map_M0_pp = pymc.MAP(M0_pp)
                        map_M0_pp.fit(method='fmin_powell')
                        l0_pred2 = get_likelihood_M0(map_M0_pp, x, pred, sigma, tau)

                        # Fit the test data using hypothesis 1 and calculate a likelihood
                        M1_pp = pymcmodels.Log_splwc_GaussianBump(x, pred, sigma)
                        map_M1_pp = pymc.MAP(M1_pp)
                        map_M1_pp.fit(method='fmin_powell')
                        l1_pred2 = get_likelihood_M1(map_M1_pp, x, pred, sigma, tau)

                    M0_logp.append(l0_pred2)
                    M1_logp.append(l1_pred2)
                    t_lrt_pred = -2 * (l0_pred2 - l1_pred2)
                    print l0_pred2, l1_pred2,  t_lrt_pred
                    t_lrt_distribution.append(t_lrt_pred)

                #
                # Plot the T_lrt distribution along with the the value of this
                # statistic

                #plt.savefig(savefig + obstype + '.t_lrt.png')

t_lrt_distribution = np.asarray(t_lrt_distribution)


pvalue = np.sum(t_lrt_distribution >= t_lrt_data) / (1.0 * nsample)

plt.figure(2)
plt.hist(t_lrt_distribution, bins=100)
plt.axvline(t_lrt_data, color='k')
plt.text(t_lrt_data, 1000, 'p=%4.3f' % (pvalue))
plt.legend()
                """
