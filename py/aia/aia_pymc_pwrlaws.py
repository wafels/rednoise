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

np.random.seed(seed=1)


#
# Functions to calculate and store results
#
def fit_summary(M, maxlogp):
    return {"AIC": M.AIC, "BIC": M.BIC, "maxlogp": maxlogp}


def T_LRT(maxlogp0, maxlogp1):
    return -2 * (maxlogp0 - maxlogp1)


# Final storage dictionary
def FitSummary(M0, l0_data, M1, l1_data):
    fitsummary = {"t_lrt": T_LRT(l0_data, l1_data),
                  "M0": fit_summary(M0, l0_data),
                  "M1": fit_summary(M1, l1_data)}
    fitsummary["dAIC"] = fitsummary["M0"]["AIC"] - fitsummary["M1"]["AIC"]
    fitsummary["dBIC"] = fitsummary["M0"]["BIC"] - fitsummary["M1"]["BIC"]
    return fitsummary


# Write out a pickle file
def pkl_write(location, fname, a):
    pkl_file_location = os.path.join(location, fname)
    print('Writing ' + pkl_file_location)
    pkl_file = open(pkl_file_location, 'wb')
    for element in a:
        pickle.dump(element, pkl_file)
    pkl_file.close()


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


def write_plots(M, region_id, obstype, savefig, model,
                bins=100, savetype='.png', extra_factors=None):
    if model == 'M0':
        variables = ['power_law_norm', 'power_law_index', 'background']
    if model == 'M1':
        variables = ['power_law_norm', 'power_law_index', 'background',
                 'gaussian_amplitude', 'gaussian_position', 'gaussian_width']

    s = M.stats()
    fontsize = 10
    alpha = 0.75
    facecolor = 'white'
    bbox = dict(facecolor=facecolor, alpha=alpha)

    for v in variables:
        # Fix the data, and get better labels
        data, mean, v95_lo, v95_hi, vlabel = fix_variables(v,
                                                           M.trace(v)[:],
                                                           s[v],
                                                           extra_factors=extra_factors)

        # Save file name
        info = '_%s_%s' % (model, v)
        filename = savefig + obstype + info + savetype

        # Find the histogram
        h, _ = np.histogram(data, bins)

        # Y position for labels
        ypos = np.max(h)

        # Plot the histogram and label it
        plt.hist(data, bins=bins)
        plt.xlabel(v + vlabel)
        plt.ylabel('p.d.f.')
        plt.title(region_id + ' ' + obstype + ' %s' % (model))

        # Summary statistics of the measurement
        plt.axvline(mean, color='k', label='mean', linewidth=2)
        plt.axvline(v95_lo, color='k', linestyle='-.', linewidth=2, label='95%, lower')
        plt.axvline(v95_hi, color='k', linestyle='--', linewidth=2, label='95%, upper')

        plt.text(mean, 0.90 * ypos, ' %f' % (mean), fontsize=fontsize, bbox=bbox)
        plt.text(v95_lo, 0.25 * ypos, ' %f' % (v95_lo), fontsize=fontsize, bbox=bbox)
        plt.text(v95_hi, 0.75 * ypos, ' %f' % (v95_hi), fontsize=fontsize, bbox=bbox)

        # Define the legend
        plt.legend(framealpha=alpha, fontsize=fontsize, loc=3)
        plt.savefig(filename)
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
            p_in = p0 + 0.1 * np.random.random(size=(6))
            n_attempt = n_attempt + 1
            print 'Attempt number: %i' % (n_attempt)
    if success:
        return answer[0]
    else:
        return p0


def get_spectrum_M1(x, A):
    return rnspectralmodels.Log_splwc_GaussianBump(x, A)


def get_likelihood_M1(map_M1, x, pwr, sigma, tau, obstype):
    A1 = get_variables_M1(map_M1)
    A1 = curve_fit_M1(x, pwr, A1, sigma)
    if obstype == '.logiobs':
        return pymc.normal_like(pwr, get_spectrum_M1(x, A1), tau)
    else:
        return pymc.lognormal_like(pwr, get_spectrum_M1(x, A1), tau)


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

    mean = s['mean']
    v95_lo = s['95% HPD interval'][0]
    v95_hi = s['95% HPD interval'][1]

    if variable == 'power_law_norm' or variable == 'background':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), ' $[\log_{10}(power)]$'

    if variable == 'gaussian_amplitude':
        return f2(data), f2(mean), f2(v95_lo), f2(v95_hi), ' $[\log_{10}(power)]$'

    if variable == 'gaussian_position':
        return f1(data, norm=extra_factors[0]), f1(mean, norm=extra_factors[0]), f1(v95_lo, norm=extra_factors[0]), f1(v95_hi, norm=extra_factors[0]), ' $[\log_{10}(frequency)]$'

    if variable == 'gaussian_width':
        return f1(data), f1(mean), f1(v95_lo), f1(v95_hi), ' [decades of frequency]'

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
regions = ['moss', 'loopfootpoints', 'sunspot', 'qs']
windows = ['hanning']
manip = 'relative'

#
# Number of posterior predictive samples to calulcate
#
nsample = 500

# PyMC control
itera = 100000
burn = 20000
thin = 5

# Set the Gaussian width for the data
sigma = 0.5 * np.log(10.0)
tau = 1.0 / (sigma ** 2)


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

            #for obstype in ['.iobs', '.logiobs']:
            for obstype in ['.logiobs']:
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
                dbname0 = os.path.join(pkl_location, 'MCMC.M0.' + region_id + obstype)
                M0 = pymc.MCMC(pymcmodel0, db='pickle', dbname=dbname0)

                # Run the sampler
                print('M0: Running simple power law model')
                M0.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)
                M0.db.close()

                # Now run the MAP
                map_M0 = pymc.MAP(pymcmodel0)
                print('M0: fitting to find the maximum likelihood')
                map_M0.fit(method='fmin_powell')

                # Get the variables and the best fit
                A0 = get_variables_M0(map_M0)
                M0_bf = get_spectrum_M0(x, A0)

                # Plot the traces
                write_plots(M0, region_id, obstype, savefig, 'M0', bins=40, extra_factors=[xnorm])

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
                dbname1 = os.path.join(pkl_location, 'MCMC.M1.'+ region_id + obstype)
                M1 = pymc.MCMC(pymcmodel1, db='pickle', dbname=dbname1)

                # Run the sampler
                print ' '
                print('M1: Running power law plus Gaussian model')
                M1.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)
                M1.db.close()

                # Now run the MAP
                map_M1 = pymc.MAP(pymcmodel1)
                print('M1: fitting to find the maximum likelihood')
                map_M1.fit(method='fmin_powell')

                # Plot the traces
                write_plots(M1, region_id, obstype, savefig, 'M1', bins=40, extra_factors=[xnorm])

                # Get the likelihood at the maximum
                likelihood1_data = map_M1.logp_at_max

                # Get the best fit spectrum under M1
                A1 = get_variables_M1(map_M1)
                M1_bf = get_spectrum_M1(x, A1)

                # Fit using M1 assuming the Gaussian has zero amplitude
                A1[3] = 0.0
                A1 = curve_fit_M1(x, pwr, A1, sigma)
                l1_data = get_likelihood_M1(map_M1, x, pwr, sigma, tau, obstype)

                # Store these results
                fitsummarydata = FitSummary(map_M0, l0_data, map_M1, l1_data)

                # Print out these results
                print ' '
                print('M0 likelihood = %f, M1 likelihood = %f' % (fitsummarydata["M0"]["maxlogp"], fitsummarydata["M1"]["maxlogp"]))
                print 'T_lrt for the initial data = %f' % (fitsummarydata["t_lrt"])
                print 'AIC_{0} - AIC_{1} = %f' % (fitsummarydata["dAIC"])
                print 'BIC_{0} - BIC_{1} = %f' % (fitsummarydata["dBIC"])

                # Save the fit summaries

                # Plot
                plt.figure(1)
                plt.loglog(freqs, np.exp(pwr_ff), label='data', color='k')

                # Plot the power law fit
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['mean']), label='M0: power law, average', color = 'b')
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='M0: power law, 95% low', color = 'b', linestyle='-.')
                plt.loglog(freqs, np.exp(M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='M0: power law, 95% high', color = 'b', linestyle='--')
                plt.loglog(freqs, np.exp(M0_bf), label='M0: power law, best fit', color = 'b', linestyle=':')

                # Plot the power law and bump fit
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['mean']), label='M1: power law + Gaussian, average', color = 'r')
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0]), label='M1: power law + Gaussian, 95% low', color = 'r', linestyle='-.')
                plt.loglog(freqs, np.exp(M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1]), label='M1: power law + Gaussian, 95% high', color = 'r', linestyle='--')
                plt.loglog(freqs, np.exp(M1_bf), label='M1: power law + Gaussian, best fit', color = 'r', linestyle=':')

                # Plot the fitness criteria - should really put this in a separate function
                ypos = np.zeros((3))
                yrange = np.max(pwr) - np.min(pwr)
                ypos_min = np.min(pwr) + 0.1 * yrange
                for yyy in range(0, ypos.size):
                    ypos[yyy] = ypos_min + yyy * (yrange - ypos_min) / (1.0 * (np.size(ypos)))
                print ypos
                plt.text(0.01, np.exp(ypos[0]), '$AIC_{0} - AIC_{1}$ = %f' % (fitsummarydata["dAIC"]))
                plt.text(0.01, np.exp(ypos[1]), '$BIC_{0} - BIC_{1}$ = %f' % (fitsummarydata["dBIC"]))
                plt.text(0.01, np.exp(ypos[2]), '$T_{LRT}$ = %f' % (fitsummarydata["t_lrt"]))

                # Complete the plot and save it
                plt.legend(loc=3, framealpha=0.5, fontsize=8)
                plt.xlabel('frequency (Hz)')
                plt.ylabel('power')
                plt.title(obstype)
                plt.savefig(savefig + obstype + '.model_fit_compare.pymc.png')
                plt.close('all')

                #
                # Do the posterior predictive comparison
                #
                # Results Storage
                good_results = []
                bad_results = []

                # Number of successful comparisons
                nfound = 0

                # Number of posterior samples available
                ntrace = np.size(M0.trace('power_law_index')[:])

                # main loop
                while nfound < nsample:

                    rthis = np.random.randint(0, high=ntrace)
                    print ' '
                    print nfound, nsample, rthis, ' (Positive T_lrt numbers favor hypothesis 1 over 0)'
                    if obstype == '.logiobs':
                        # Generate test data under hypothesis 0
                        pred = M0.trace('predictive')[rthis]

                        # Fit the test data using hypothesis 0 and calculate a likelihood
                        M0_pp = pymcmodels.Log_splwc(x, pred, sigma)
                        map_M0_pp = pymc.MAP(M0_pp)
                        try:
                            print('M0: fitting to find the maximum likelihood')
                            map_M0_pp.fit(method='fmin_powell')
                            l0 = get_likelihood_M0(map_M0_pp, x, pred, sigma, tau, obstype)
                        except:
                            print('Error fitting M0 to sample.')
                            l0 = None

                        # Fit the test data using hypothesis 1 and calculate a likelihood
                        M1_pp = pymcmodels.Log_splwc_GaussianBump(x, pred, sigma)
                        map_M1_pp = pymc.MAP(M1_pp)
                        try:
                            print('M1: fitting to find the maximum likelihood')
                            map_M1_pp.fit(method='fmin_powell')
                            l1 = get_likelihood_M1(map_M1_pp, x, pred, sigma, tau, obstype)
                        except:
                            print('Error fitting M1 to sample.')
                            l1 = None

                    # Sort the results into good and bad.
                    if (l0 != None) and (l1 != None):
                        t_lrt_pred = T_LRT(l0, l1)
                        if t_lrt_pred >= 0.0:
                            good_results.append((rthis,
                                                 FitSummary(map_M0_pp, l0,
                                                            map_M1_pp, l1)))
                            nfound = nfound + 1
                            print('    T_lrt = %f' % (good_results[-1][1]['t_lrt']))
                        else:
                            print('! T_lrt = %f <0 ! - must be a bad fit' % (t_lrt_pred))
                            bad_results.append((rthis,
                                                FitSummary(map_M0_pp, l0,
                                                           map_M1_pp, l1)))
                    else:
                        print('! Fitting algorithm fell down !')
                        bad_results.append((rthis,
                                            FitSummary(map_M0_pp, l0,
                                                       map_M1_pp, l1)))

                # Save the good and bad results and use a different program to plot the results
                pkl_write(pkl_location,
                          'posterior_predictive.' + region_id + '.good.pickle',
                          good_results)

                pkl_write(pkl_location,
                          'posterior_predictive.' + region_id + '.bad.pickle',
                          bad_results)

                # Quick summary plots
                t_lrt_data = fitsummarydata["t_lrt"]
                t_lrt_distribution = np.asarray([z[1]["t_lrt"] for z in good_results])

                pvalue = np.sum(t_lrt_distribution >= t_lrt_data) / (1.0 * nsample)
                print('Posterior predictive p-value = %f' % (pvalue))

                plt.figure(2)
                plt.hist(t_lrt_distribution, bins=100)
                plt.axvline(t_lrt_data, color='k', label='p=%4.3f' % (pvalue))
                plt.legend()
                plt.savefig(savefig + obstype + '.posterior_predictive.png')
