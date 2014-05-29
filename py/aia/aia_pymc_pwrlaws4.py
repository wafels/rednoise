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
    A = [m.value.power_law_norm, m.value.power_law_index, m.value.background,
         m.value.nfactor]
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
        variables = ['power_law_norm', 'power_law_index', 'background', 'nfactor']
    if model == 'M1':
        variables = ['power_law_norm', 'power_law_index', 'background',
                 'gaussian_amplitude', 'gaussian_position', 'gaussian_width', 'nfactor']
    if model == 'M2':
        variables = ['power_law_norm', 'power_law_index', 'background',
                 'gaussian_amplitude', 'gaussian_position', 'gaussian_width', 'nfactor']

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
         m.value.gaussian_width, m.value.nfactor]
    return A


def curve_fit_M1(x, y, p0, sigma, n_attempt_limit=10):
    p_in = p0
    n_attempt = 0
    success = False
    while (not success) and (n_attempt <= n_attempt_limit):
        try:
            answer = curve_fit(rnspectralmodels.Log_splwc_AddNormalBump2_CF, x, y, p0=p_in, sigma=sigma)
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
    return rnspectralmodels.Log_splwc_AddNormalBump2(x, A)


def get_likelihood_M1(map_M1, x, pwr, sigma, obstype):
    tau = 1.0 / (sigma ** 2)
    A1 = get_variables_M1(map_M1)[0:6]
    A1 = curve_fit_M1(x, pwr, A1, sigma)
    return pymc.normal_like(pwr, get_spectrum_M1(x, A1), tau)


def get_chi_M1(map_M1, x, pwr, sigma):
    A1 = get_variables_M1(map_M1)
    spectrum = get_spectrum_M1(x, A1)
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


plt.ioff()
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_final/'
sfigroot = '~/ts/img_cc_final/'
scsvroot = '~/ts/csv_cc_final/'
corename = 'shutdownfun3_6hr'
sunlocation = 'disk'
fits_level = '1.5'
waves = ['171', '193']
regions = ['sunspot', 'qs', 'moss', 'loopfootpoints']
windows = ['hanning']
manip = 'relative'

"""
#
# Limb
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle/'
sfigroot = '~/ts/img/'
scsvroot = '~/ts/csv/'
corename = 'shutdownfun6_6hr'
sunlocation = 'limb'
fits_level = '1.0'
waves = ['171']
regions = ['highlimb', 'crosslimb', 'lowlimb', 'moss', 'loopfootpoints1', 'loopfootpoints2']
windows = ['hanning']
manip = 'relative'
"""
#
# Number of posterior predictive samples to calculate
#
# Set to 1 to do a full analytical run.  Set to a high number to do a test
# run
testing = 100

nsample = 5000 / testing

# PyMC control
itera = 100000 # / testing
burn = 20000 #/ testing
thin = 5

# Image file type
imgfiletype = 'pdf'

# Frequency scaling
freqfactor = 1000.0

for iwave, wave in enumerate(waves):
    print(' ')
    print('##################################################################')
    prettyprint('Data')
    # Which wavelength?
    print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))

    # Now that the loading and saving locations are seot up, proceed with
    # the analysis.
    for iregion, region in enumerate(regions):
        # Which region
        prettyprint('Region')
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
                # Frequencies
                freqs = pickle.load(pkl_file)
                # Averaged Fourier Power per frequency
                pwr_average = pickle.load(pkl_file)
                # Peak Fourier power per frequency
                pwr_ff_peak = pickle.load(pkl_file)
                # Fitted Peak of the Fourier power per frequency
                pwr_ff = pickle.load(pkl_file)
                # Number of pixels used to create the average
                npixels = pickle.load(pkl_file)
                # Lag zero cross correlation coefficient
                ccc0 = pickle.load(pkl_file)
                # Lag 'lag' coefficient
                lag = pickle.load(pkl_file)
                ccclag = pickle.load(pkl_file)
                # Pixel distance of the cross correlation function
                #xccc = pickle.load(pkl_file)
                # Cross correlation function
                #ccc = pickle.load(pkl_file)
                # Decay constant for the cross-correlation function
                #ccc_answer = pickle.load(pkl_file)
                pkl_file.close()
                # Load the widths
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.distribution_widths.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                fitted_width = pickle.load(pkl_file)
                std_dev = pickle.load(pkl_file)
                abs_diff = pickle.load(pkl_file)
                pkl_file.close()
                # Load the coherence
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.coherence.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                freqs = pickle.load(pkl_file)
                coher = pickle.load(pkl_file)
                coher_max = pickle.load(pkl_file)
                coher_mode = pickle.load(pkl_file)
                coher_95_hi = pickle.load(pkl_file)
                coher_mean = pickle.load(pkl_file)
                pkl_file.close()

                # Something wrong with the last coherence - estimate it
                coher_95_hi[-1] = coher_95_hi[-2]
                coher_mean[-1] = coher_mean[-2]

                # Load the correlative measures
                ifilename = 'OUT.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.correlative.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                cc0_ds = pickle.load(pkl_file)
                cclag_ds = pickle.load(pkl_file)
                ccmax_ds = pickle.load(pkl_file)
                pkl_file.close()

                # Find out the length-scale at which the fitted
                # cross-correlation coefficient reaches the value of 0.1
                #decorr_length = ccc_answer[1] * (np.log(ccc_answer[0]) - np.log(0.1))
                #print("Decorrelation length = %f pixels" % (decorr_length))

                # Fix any non-finite widths and divide by the number of pixels
                # Note that this should be the effective number of pixels,
                # given that the original pixels are not statistically separate.
                prettyprint('Pixel independence calculation')
                print 'Number of pixels ', npixels
                print 'Average lag 0 cross correlation coefficient = %f' % (ccc0)
                print 'Average lag %i cross correlation coefficient = %f' % (lag, ccclag)
                # Independence coefficient - how independent is one pixel compared
                # to its nearest neighbour?
                # independence_coefficient = 1.0 - 0.5 * (np.abs(ccc0) + np.abs(ccclag))
                #
                # NOTE: The astroML package has a method for sampling from an
                # empirical (1-d) distribution.  See http://www.astroml.org/book_figures/chapter3/fig_clone_distribution.html
                # or use the following code snippet:
                # from astroML.density_estimation import EmpiricalDistribution
                # # generate an 'observed' bimodal distribution with 10000 values
                # dists = (stats.norm(-1.3, 0.5), stats.norm(1.3, 0.5))
                # fracs = (0.6, 0.4)
                # x = np.hstack((d.rvs(f * Ndata) for d, f in zip(dists, fracs)))
                # # We can clone the distribution easily with this function
                # x_cloned = EmpiricalDistribution(x).rvs(Nclone)
                #
                # Could use this to sample from the empirical distribution of
                # the independence distribution.  Hmmm - would this work with
                # PyMC?
                #
                #independence_coefficient = 1.0 - np.abs(ccc0)

                # Use the coherence estimate to get a frequency-dependent
                # independence coefficient
                # independence_coefficient = 1.0 - coher
                independence_coefficient = 1.0 - np.abs(ccc0)
                print 'Average independence coefficient ', np.mean(independence_coefficient)
                npixels_effective = independence_coefficient * (npixels - 1) + 1
                print("Average effective number of independent observations = %f " % (np.mean(npixels_effective)))
                sigma_of_distribution = fix_nonfinite(std_dev)
                sigma_for_mean = sigma_of_distribution #/ np.sqrt(npixels_effective)

                # Frequency-dependent independence coefficient
                #independence_coefficient = 1.0 - np.abs(ccc0)
                #print 'Coherence: Average independence coefficient ', np.mean(independence_coefficient)
                #npixels_effective = independence_coefficient * (npixels - 1) + 1
                #print("Coherence: Effective number of independent observations = %f " % (np.mean(npixels_effective)))
                #sigma_for_mean = sigma_of_distribution / np.sqrt(npixels_effective)
                #unbiased_factor = np.sqrt(npixels_effective / (npixels_effective ** 2 - npixels_effective))
                #print("Unbiased factor = %f" % (unbiased_factor))
                #sigma = unbiased_factor * fix_nonfinite(sigma)

                # Some models fitting use tau instead of sigma
                #tau = 1.0 / (sigma ** 2)

                # Normalize the frequency
                xnorm = freqs[0]
                x = freqs / xnorm
                nfreq = freqs.size

                #
                # Model 0 - the power law
                #
                # Two passes - first pass to get an initial estimate of the
                # spectrum, and a second to get a better estimate using the
                # sigma mean
                prettyprint('Model fitting and results')
                sigma = sigma_of_distribution
                tau = 1.0 / (sigma ** 2)
                pwr = pwr_ff
                pymcmodel0 = pymcmodels3.Log_splwc(x, pwr, sigma)
                jjj = 0
                passnumber = str(jjj)

                # Initiate the sampler
                dbname0 = os.path.join(pkl_location, 'MCMC.M0.' + passnumber + region_id + obstype)
                M0 = pymc.MCMC(pymcmodel0, db='pickle', dbname=dbname0)

                # Run the sampler
                print('M0 : pass %i : Running simple power law model' % (jjj + 1))
                M0.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)
                M0.db.close()

                # Now run the MAP
                map_M0 = pymc.MAP(pymcmodel0)
                print(' ')
                print('M0 : pass %i : fitting to find the maximum likelihood' % (jjj + 1))
                map_M0.fit(method='fmin_powell')

                # Get the variables and the best fit
                A0 = get_variables_M0(map_M0)
                M0_bf = get_spectrum_M0(x, A0)

                # Plot the traces
                write_plots(M0, region_id, obstype, savefig, 'M0', passnumber, bins=40, extra_factors=[xnorm])

                # Get the likelihood at the maximum
                #likelihood0_data = map_M0.logp_at_max
                l0_data = get_likelihood_M0(map_M0, x, pwr, sigma / np.sqrt(A0[3]), obstype)
                l0_data_logp = get_log_likelihood(pwr, M0_bf, sigma / np.sqrt(A0[3]))

                # Get the chi-squared value
                chi0 = get_chi_M0(map_M0, x, pwr, sigma / np.sqrt(A0[3]))
                print('M0 : pass %i : reduced chi-squared = %f' % (jjj + 1, chi0))
                print('   : variables ', A0)
                print('   : estimate log likelihood = %f' % (l0_data_logp))

                #
                # Get an estimate of where the Gaussian bump might be for the
                # second model
                #
                # Frequency range we will consider for the bump
                physical_bump_frequency_limits = np.asarray([10.0 ** -3.5, 10.0 ** -2.0])
                bump_frequency_limits = physical_bump_frequency_limits / xnorm
                log_bump_frequency_limits = np.log(bump_frequency_limits)
                bump_loc_lo = x >= bump_frequency_limits[0]
                bump_loc_hi = x <= bump_frequency_limits[1]
                # Get where there is excess unexplained power
                positive_difference = pwr - M0_bf > 0
                # Combine all the conditions to get where the bump might be
                bump_loc = positive_difference * bump_loc_lo * bump_loc_hi

                # If there are sufficient points, get an estimate of the bump
                if bump_loc.sum() >= 10:
                    g_estimate, _ = curve_fit(rnspectralmodels.NormalBump2_CF, np.log(x[bump_loc]), (pwr - M0_bf)[bump_loc])
                    A1_estimate = [A0[0], A0[1], A0[2], g_estimate[0], g_estimate[1], g_estimate[2]]
                else:
                    A1_estimate = None
                print "Second model estimate : ", A1_estimate
                print "Physical bump frequency position limits : ", physical_bump_frequency_limits
                print "Normalized log bump frequency limits : ", log_bump_frequency_limits

                #
                # Second model - power law with Gaussian bumps
                #
                sigma = sigma_of_distribution
                tau = 1.0 / (sigma ** 2)
                pwr = pwr_ff
                if A1_estimate[4] < log_bump_frequency_limits[0]:
                    A1_estimate[4] = log_bump_frequency_limits[0]
                    print '!! Estimate fit exceeded lower limit of log_bump_frequency_limits. Resetting to limit'
                if A1_estimate[4] > log_bump_frequency_limits[1]:
                    A1_estimate[4] = log_bump_frequency_limits[1]
                    print '!! Estimate fit exceeded upper limit of log_bump_frequency_limits. Resetting to limit'
                pymcmodel1 = pymcmodels3.Log_splwc_AddNormalBump2(x, pwr, sigma, init=A1_estimate, log_bump_frequency_limits=log_bump_frequency_limits)

                # Define the pass number
                jjj = 0
                passnumber = str(jjj)

                # Initiate the sampler
                dbname1 = os.path.join(pkl_location, 'MCMC.M1.' + passnumber + region_id + obstype)
                M1 = pymc.MCMC(pymcmodel1, db='pickle', dbname=dbname1)

                # Run the sampler
                print('M1 : pass %i : Running power law plus bump model' % (jjj + 1))
                M1.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)
                M1.db.close()

                # Now run the MAP
                map_M1 = pymc.MAP(pymcmodel1)
                print(' ')
                print('M1 : pass %i : fitting to find the maximum likelihood' % (jjj + 1))
                map_M1.fit(method='fmin_powell')

                # Get the variables and the best fit
                A1 = get_variables_M1(map_M1)
                M1_bf = get_spectrum_M1(x, A1)

                # Plot the traces
                write_plots(M1, region_id, obstype, savefig, 'M1', passnumber, bins=40, extra_factors=[xnorm])

                # Get the likelihood at the maximum
                #likelihood1_data = map_M1.logp_at_max
                l1_data = get_likelihood_M1(map_M1, x, pwr, sigma / np.sqrt(A1[6]), obstype)
                l1_data_logp = get_log_likelihood(pwr, M1_bf, sigma / np.sqrt(A1[6]))

                # Get the chi-squared value
                chi1 = get_chi_M1(map_M1, x, pwr, sigma / np.sqrt(A1[6]))
                print('M1 : pass %i : reduced chi-squared = %f' % (jjj + 1, chi1))
                print('   : variables ', A1)
                print('   : estimate log likelihood = %f' % (l1_data_logp))

                # Store these results
                fitsummarydata = FitSummary(map_M0, l0_data, chi0, map_M1, l1_data, chi1)

                # Print out these results
                print ' '
                print('M0 likelihood = %f, M1 likelihood = %f' % (fitsummarydata["M0"]["maxlogp"], fitsummarydata["M1"]["maxlogp"]))
                print('M0 likelihood (explicit calculation) = %f, M1 likelihood (explicit calculation) = %f' % (l0_data_logp, l1_data_logp))
                print 'T_lrt for the initial data = %f' % (fitsummarydata["t_lrt"])
                print 'T_lrt for the initial data (explicit calculation) = %f' % (T_LRT(l0_data_logp, l1_data_logp))
                print 'AIC_{0} - AIC_{1} = %f' % (fitsummarydata["dAIC"])
                print 'BIC_{0} - BIC_{1} = %f' % (fitsummarydata["dBIC"])
                print 'M0: chi-squared = %f' % (fitsummarydata["M0"]["chi2"])
                print 'M1: chi-squared = %f' % (fitsummarydata["M1"]["chi2"])

                # Calculate the best fit Gaussian contribution
                normalbump_BF = np.log(rnspectralmodels.NormalBump2_CF(np.log(x), A1[3], A1[4], A1[5]))
                # Calculate the best fit power law contribution
                powerlaw_BF = np.log(rnspectralmodels.power_law_with_constant(x, A1[0:3]))

                # Plot
                title = 'AIA ' + wave + " : " + sunday_name[region]
                xvalue = freqfactor * freqs
                fivemin = freqfactor * 1.0 / 300.0
                threemin = freqfactor * 1.0 / 180.0

                # -------------------------------------------------------------
                # Plot the data and the model fits
                ax = plt.subplot(111)

                # Set the scale type on each axis
                ax.set_xscale('log')
                ax.set_yscale('log')

                # Set the formatting of the tick labels
                xformatter = plt.FuncFormatter(log_10_product)
                ax.xaxis.set_major_formatter(xformatter)

                ax.plot(xvalue, np.exp(pwr_ff), label='data', color='k')

                # Plot the M0 fit
                chired = '\chi^{2}_{red}'
                chiformat = '%4.2f'
                chivalue = chiformat % (fitsummarydata["M0"]["chi2"])
                chivaluestring = '$' + chired + '=' + chivalue + '$'
                ax.plot(xvalue, np.exp(M0_bf), label='$M_{1}$: maximum likelihood fit, ' + chivaluestring, color='b')
                ax.plot(xvalue, np.exp(M0.stats()['fourier_power_spectrum']['mean']), label='$M_{1}$: posterior mean', color = 'b', linestyle='--')
                #plt.plot(xvalue, M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0], label='M0: 95% low', color = 'b', linestyle='-.')
                #plt.plot(xvalue, M0.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1], label='M0: 95% high', color = 'b', linestyle='--')

                # Plot the M1 fit
                chivalue = chiformat % (fitsummarydata["M1"]["chi2"])
                chivaluestring = '$' + chired + '=' + chivalue + '$'
                ax.plot(xvalue, np.exp(M1_bf), label='$M_{2}$: maximum likelihood fit, ' + chivaluestring, color='r')
                ax.plot(xvalue, np.exp(M1.stats()['fourier_power_spectrum']['mean']), label='$M_{2}$: posterior mean', color = 'r', linestyle='--')
                #plt.plot(xvalue, M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 0], label='M1: 95% low', color = 'r', linestyle='-.')
                #plt.plot(xvalue, M1.stats()['fourier_power_spectrum']['95% HPD interval'][:, 1], label='M1: 95% high', color = 'r', linestyle='--')

                # Plot each component of M1
                ax.plot(xvalue, np.exp(powerlaw_BF), label='power law component of maximum likelihood fit, $M_{2}$', color='g')
                ax.plot(xvalue, np.exp(normalbump_BF), label='Gaussian component of maximum likelihood fit, $M_{2}$', color='g', linestyle='--')

                # Plot the 3 and 5 minute frequencies
                ax.axvline(fivemin, label='5 minutes', linestyle='--', color='k')
                ax.axvline(threemin, label='3 minutes', linestyle='-.', color='k')

                # Plot the bump limits
                #plt.axvline(np.log10(physical_bump_frequency_limits[0]), label='bump center position limit', linestyle=':' ,color='k')
                #plt.axvline(np.log10(physical_bump_frequency_limits[1]), linestyle=':' ,color='k')

                # Plot the fitness coefficients
                # Plot the fitness criteria - should really put this in a separate function
                xpos = freqfactor * 10.0 ** -2.0
                ypos = np.zeros((2))
                ypos_max = np.max(pwr_ff)
                ypos_min = np.min(pwr_ff) + 0.5 * (np.max(pwr_ff) - np.min(pwr_ff))
                yrange = ypos_max - ypos_min
                for yyy in range(0, ypos.size):
                    ypos[yyy] = ypos_min + yyy * yrange / (1.0 * (np.size(ypos) - 1.0))
                #plt.text(xpos, ypos[0], '$AIC_{0} - AIC_{1}$ = %f' % (fitsummarydata["dAIC"]))
                #plt.text(xpos, ypos[1], '$BIC_{0} - BIC_{1}$ = %f' % (fitsummarydata["dBIC"]))
                #plt.text(xpos, ypos[2], '$T_{LRT}$ = %f' % (fitsummarydata["t_lrt"]))
                #plt.text(xpos, ypos[0], '$M_{1}$: reduced $\chi^{2}$ = %f' % (fitsummarydata["M0"]["chi2"]))
                #plt.text(xpos, ypos[1], '$M_{2}$: reduced $\chi^{2}$ = %f' % (fitsummarydata["M1"]["chi2"]))

                # Complete the plot and save it
                plt.legend(framealpha=0.5, fontsize=8, labelspacing=0.2, loc=3)
                plt.xlabel('frequency (mHz)')
                plt.ylabel('power')
                plt.title(title)
                ymin_plotted = np.exp(np.asarray([np.min(pwr_ff) - 1.0,
                                           np.max(normalbump_BF) - 2.0]))
                #plt.ylim(np.min(ymin_plotted), np.exp(np.max(pwr_ff) + 1.0))
                plt.ylim(fit_details()['ylim'][0], fit_details()['ylim'][1])
                plt.savefig(savefig + obstype + '.' + passnumber + '.model_fit_compare.pymc.%s' % (imgfiletype))
                plt.close('all')

                # -------------------------------------------------------------
                # Measures of the residuals
                plt.figure(2)
                plt.plot(xvalue, pwr_ff - M0_bf, label='data - M0')
                plt.plot(xvalue, pwr_ff - M1_bf, label='data - M1')
                plt.plot(xvalue, np.abs(pwr_ff - M0_bf), label='|data - M0|')
                plt.plot(xvalue, np.abs(pwr_ff - M1_bf), label='|data - M1|')
                plt.plot(xvalue, sigma_of_distribution, label='estimated log(power) distribution width')
                plt.plot(xvalue, sigma_for_mean, label='estimated error in mean')
                plt.axhline(0.0, color='k')

                # Plot the 3 and 5 minute frequencies
                plt.axvline(fivemin, label='5 mins.', linestyle='--', color='k')
                plt.axvline(threemin, label='3 mins.', linestyle='-.', color='k' )

                # Plot the bump limits
                plt.axvline(np.log10(physical_bump_frequency_limits[0]), label='bump center position limit', linestyle=':' ,color='k')
                plt.axvline(np.log10(physical_bump_frequency_limits[1]), linestyle=':' ,color='k')

                # Complete the plot and save it
                plt.legend(framealpha=0.5, fontsize=8)
                plt.xlabel('log10(frequency (Hz))')
                plt.ylabel('fit residuals')
                plt.title(title)
                plt.savefig(savefig + obstype + '.' + passnumber + '.model_fit_residuals.pymc.png')
                plt.close('all')

                # -------------------------------------------------------------
                # Power ratio
                plt.figure(2)
                plt.semilogy(xvalue, np.exp(normalbump_BF) / np.exp(powerlaw_BF), label='M1[normal bump] / M1[power law]')
                plt.semilogy(xvalue, np.exp(pwr_ff) / np.exp(M1_bf), label='data / M1 best fit')
                plt.semilogy(xvalue, np.exp(pwr_ff) / np.exp(normalbump_BF), label='data / M1[normal bump]')
                plt.semilogy(xvalue, np.exp(pwr_ff) / np.exp(powerlaw_BF), label='data / M1[power law]')
                plt.axhline(1.0, color='k')
                plt.ylim(0.01, 100.0)

                # Plot the 3 and 5 minute frequencies
                plt.axvline(fivemin, label='5 mins.', linestyle='--', color='k')
                plt.axvline(threemin, label='3 mins.', linestyle='-.', color='k')

                # Plot the bump limits
                plt.axvline(np.log10(physical_bump_frequency_limits[0]), label='bump center position limit', linestyle=':' ,color='k')
                plt.axvline(np.log10(physical_bump_frequency_limits[1]), linestyle=':' ,color='k')

                # Complete the plot
                plt.xlabel('log10(frequency (Hz))')
                plt.ylabel('ratio')
                plt.legend(framealpha=0.5, fontsize=8)
                plt.title(title)
                plt.savefig(savefig + obstype + '.' + passnumber + '.model_fit_ratios.pymc.png')
                plt.close('all')

                #--------------------------------------------------------------
                # Residuals / estimated noise - these are the contributions to
                # chi-squared.  Also calculate the Shapiro-Wilks test and the
                # Anderson-Darling test for normality of the M1_bf residuals.
                # These will let us know if the reduced chi-squared values
                # are OK to use.
                M1_residuals = pwr_ff - M1_bf
                nmzd = M1_residuals / sigma_for_mean
                anderson_result = anderson(nmzd, dist='norm')
                ad_crit = anderson_result[1]
                ad_sig = anderson_result[2]
                for jjj in range(0, 5):
                    if anderson_result[0] >= ad_crit[jjj]:
                        anderson_level = ad_sig[jjj]

                shapiro_result = shapiro(nmzd)
                plt.figure(2)
                plt.semilogx(xvalue, (pwr_ff - M0_bf) / sigma_for_mean, label='(data - M0) / sigma for mean')
                plt.semilogx(xvalue, nmzd, label='(data - M1) / sigma for mean')
                plt.axhline(0.0, color='k')

                # Print the normality tests to the plot
                plt.text(xvalue[0], 0.01, 'M1: Shapiro-Wilks p=%f' % (shapiro_result[1]))
                plt.text(xvalue[0], 0.10, 'M1: Anderson-Darling: reject above %s %% (AD=%f)' % (str(anderson_level), anderson_result[0]))

                # Plot the 3 and 5 minute frequencies
                plt.axvline(fivemin, label='5 mins.', linestyle='--', color='k')
                plt.axvline(threemin, label='3 mins.', linestyle='-.', color='k')

                # Plot the bump limits
                plt.axvline(physical_bump_frequency_limits[0], label='bump center position limit', linestyle=':' ,color='k')
                plt.axvline(physical_bump_frequency_limits[1], linestyle=':' ,color='k')

                # Complete the plot and save it
                plt.legend(framealpha=0.5, fontsize=8)
                plt.xlabel('log10(frequency (Hz))')
                plt.ylabel('scaled residual')
                plt.title(title)
                plt.savefig(savefig + obstype + '.' + passnumber + '.scaled_residual.png')
                plt.close('all')

                #--------------------------------------------------------------
                # qq-plot
                qqplot(nmzd.flatten(), line='45', fit=True)
                plt.savefig(savefig + obstype + '.' + passnumber + '.qqplot.png')
                plt.close('all')

                ###############################################################
                # Save the model fitting information
                pkl_write(pkl_location,
                      'M0_maximum_likelihood.' + region_id + '.pickle',
                      (A0, fitsummarydata, pwr, sigma, M0_bf))

                pkl_write(pkl_location,
                      'M1_maximum_likelihood.' + region_id + '.pickle',
                      (A1, fitsummarydata, pwr, sigma, M1_bf))

                # M1 residuals - saved as CSV so they can be used by other
                # analysis environments
                csv_timeseries_write(os.path.join(os.path.expanduser(scsv)),
                                     '.'.join((ident, 'M1_residuals.csv')),
                                     (freqs, M1_residuals))

                # M1 estimated deviance- saved as CSV so they can be used by
                # other analysis environments
                csv_timeseries_write(os.path.join(os.path.expanduser(scsv)),
                                     '.'.join((ident, 'sigma_for_mean.csv')),
                                     (freqs, sigma_for_mean))

                ###############################################################
                # Second part - model selection through posterior predictive
                # checking.
                prettyprint('Model selection through posterior predictive checking - T_LRT')
                # Results Storage
                good_results = []
                bad_results = []

                # Number of successful comparisons
                nfound = 0

                # Number of posterior samples available
                ntrace = np.size(M0.trace('power_law_index')[:])

                # main loop
                rthis = 0
                rtried = []
                while nfound < nsample:
                    # random number
                    rthis = np.random.randint(0, high=ntrace)
                    while rthis in rtried:
                        rthis = np.random.randint(0, high=ntrace)
                    rtried.append(rthis)
                    #print ' '
                    #print nfound, nsample, rthis, ' (Positive T_lrt numbers favor hypothesis 1 over 0)'

                    # Generate test data under hypothesis 0
                    pred = M1.trace('predictive')[rthis]

                    # Fit the test data using hypothesis 0 and calculate a likelihood
                    try:
                        #
                        # Curve fits.  This should really be replaced by the
                        # full Bayesian fit as the Bayesian fit in this program
                        # includes variable noise in the form of the "nfactor"
                        # variable.  Unfortunately, this will take much much
                        # longer than the current process.
                        #
                        # Curve fits
                        # ----------
                        #fit0, _ = curve_fit(rnspectralmodels.Log_splwc_CF, x, pred, sigma=sigma / np.sqrt(A0[3]), p0=A0[0:3])
                        #fit0_bf = get_spectrum_M0(x, fit0)
                        #l0 = get_log_likelihood(pred, fit0_bf, sigma / np.sqrt(A0[3]))
                        #
                        # Use PyMC
                        # --------
                        pymcmodel0 = pymcmodels3.Log_splwc(x, pred, sigma)
                        #M0 = pymc.MCMC(pymcmodel0)
                        #M0.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)
                        map_M0 = pymc.MAP(pymcmodel0)
                        map_M0.fit(method='fmin_powell')
                        A0_bf = get_variables_M0(map_M0)
                        fit0_bf = get_spectrum_M0(x, A0_bf)
                        l0 = get_log_likelihood(pred, fit0_bf, sigma / np.sqrt(A0_bf[3]))
                        print 'l0 = %f' % (l0)
                    except:
                        print('LRT: Error fitting M0 to sample.')
                        l0 = None

                    # Fit the test data using hypothesis 1 and calculate a likelihood
                    try:
                        #
                        # Curve fits.  This should really be replaced by the
                        # full Bayesian fit as the Bayesian fit in this program
                        # includes variable noise in the form of the "nfactor"
                        # variable.  Unfortunately, this will take much much
                        # longer than the current process.
                        #
                        # Curve fits
                        # ----------
                        #fit1, _ = curve_fit(rnspectralmodels.Log_splwc_AddNormalBump2_CF, x, pred, sigma=sigma / np.sqrt(A1[6]), p0=A1[0:6])
                        #fit1_bf = get_spectrum_M1(x, fit1)
                        #l1 = get_log_likelihood(pred, fit1_bf, sigma / np.sqrt(A1[6]))
                        #
                        # Use PyMC
                        # --------
                        pymcmodel1 = pymcmodels3.Log_splwc_AddNormalBump2(x, pred, sigma, init=A1_estimate, log_bump_frequency_limits=log_bump_frequency_limits)
                        #M1 = pymc.MCMC(pymcmodel1)
                        #M1.sample(iter=itera, burn=burn, thin=thin, progress_bar=True)
                        map_M1 = pymc.MAP(pymcmodel1)
                        map_M1.fit(method='fmin_powell')
                        A1_bf = get_variables_M1(map_M1)
                        fit1_bf = get_spectrum_M1(x, A1_bf)
                        l1 = get_log_likelihood(pred, fit1_bf, sigma / np.sqrt(A1_bf[6]))
                        print 'l1 = %f' % (l1)
                    except:
                        print('LRT: Error fitting M1 to sample.')
                        l1 = None

                    # Sort the results into good and bad.
                    if (l0 != None) and (l1 != None):
                        t_lrt_pred = T_LRT(l0, l1)
                        if t_lrt_pred >= 0.0:
                            fit_results = (rthis, l0, chi0, l1, chi1, t_lrt_pred)
                            good_results.append(fit_results)
                            nfound = nfound + 1
                            print('#%i    T_lrt = %f' % (nfound, T_LRT(l0, l1)))
                            if np.mod(nfound, 1000) == 0:
                                plt.figure(1)
                                plt.plot(xvalue, pred, label='predicted', color='k')
                                plt.plot(xvalue, fit0_bf, label='M0 fit', color='b')
                                plt.plot(xvalue, fit1_bf, label='M1 fit', color='r')
                                plt.xlabel("log10(frequency)")
                                plt.ylabel("log(power)")
                                plt.legend(fontsize=8, framealpha=0.5)
                                plt.title(title + " : pp check sample #%i" % (nfound))
                                plt.savefig(savefig + obstype + '.posterior_predictive.sample.' + str(nfound) + '.png')
                                plt.close('all')
                        else:
                            #print('! T_lrt = %f <0 ! - must be a bad fit' % (t_lrt_pred))
                            bad_results.append((rthis, l0, chi0, l1, chi1, t_lrt_pred))
                    else:
                        #print('! Fitting algorithm fell down !')
                        bad_results.append((rthis, l0, chi0, l1, chi1, None))

                # Save the good and bad results and use a different program to plot the results
                pkl_write(pkl_location,
                          'posterior_predictive.T_LRT.' + region_id + '.good.pickle',
                          good_results)

                pkl_write(pkl_location,
                          'posterior_predictive.T_LRT.' + region_id + '.bad.pickle',
                          bad_results)

                # Quick summary plots
                t_lrt_data = fitsummarydata["t_lrt"]
                t_lrt_distribution = np.asarray([z[5] for z in good_results])

                pvalue = np.sum(t_lrt_distribution >= t_lrt_data) / (1.0 * nsample)
                print('Posterior predictive p-value = %f' % (pvalue))

                plt.figure(2)
                plt.hist(t_lrt_distribution, bins=30)
                plt.xlabel('LRT statistic')
                plt.ylabel('Number found [%i samples]' % (nsample))
                plt.title(title)
                plt.axvline(t_lrt_data, color='k', label='p=%4.3f [$T_{LRT}=$%f]' % (pvalue, t_lrt_data))
                plt.legend(fontsize=8)
                plt.savefig(savefig + obstype + '.posterior_predictive.T_LRT.png')
                plt.close('all')

                ###############################################################
                # Third part - model fitness through posterior predictive
                # checking.
                prettyprint('Model checking through posterior predictive checking - T_SSE')
                good_results2 = []
                bad_results2 = []

                # Number of successful comparisons
                nfound2 = 0

                # Number of posterior samples available
                ntrace = np.size(M1.trace('power_law_index')[:])

                # T_SSE for the data
                t_sse_data = T_SSE(pwr_ff, M1_bf, sigma / np.sqrt(A1[6]))
                print 'T_SSE for the data = %f' % (t_sse_data)
                # main loop
                rthis = 0
                rtried = []
                while nfound2 < nsample:
                    # random number
                    rthis = np.random.randint(0, high=ntrace)
                    while rthis in rtried:
                        rthis = np.random.randint(0, high=ntrace)
                    rtried.append(rthis)
                    #print ' '
                    #print nfound, nsample, rthis, ' (Positive T_lrt numbers favor hypothesis 1 over 0)'

                    # Generate test data under hypothesis 1
                    pred = M1.trace('predictive')[rthis]

                    # Fit the test data using hypothesis 1 and calculate a likelihood

                    try:
                        #
                        # Curve fits
                        #
                        fit1, _ = curve_fit(rnspectralmodels.Log_splwc_AddNormalBump2_CF, x, pred, sigma=sigma / np.sqrt(A1[6]), p0=A1[0:6])
                        fit1_bf = get_spectrum_M1(x, fit1)
                        t_sse_pred = T_SSE(pred, fit1_bf, sigma / np.sqrt(A1[6]))
                    except:
                        print('SSE: Error fitting M1 to sample.')
                        t_sse_pred = None

                    # Sort the results into good and bad.
                    if (t_sse_pred != None):
                        print t_sse_pred, t_sse_data
                        fit_results = (rthis, t_sse_pred)
                        good_results2.append(fit_results)
                        nfound2 = nfound2 + 1
                        if np.mod(nfound2, 1000) == 0:
                            plt.figure(1)
                            plt.plot(xvalue, pred, label='predicted', color='k')
                            plt.plot(xvalue, fit1_bf, label='M1 fit', color='r')
                            plt.xlabel("log10(frequency)")
                            plt.ylabel("log(power)")
                            plt.legend(fontsize=8, framealpha=0.5)
                            plt.title(title + " : pp check sample #%i" % (nfound2))
                            plt.savefig(savefig + obstype + '.posterior_predictive.T_SSE.sample.' + str(nfound2) + '.png')
                            plt.close('all')
                    else:
                        bad_results2.append((rthis, t_sse_pred))

                # Save the good and bad results and use a different program to plot the results
                pkl_write(pkl_location,
                          'posterior_predictive.T_SSE.' + region_id + '.good.pickle',
                          good_results2)

                pkl_write(pkl_location,
                          'posterior_predictive.T_SSE.' + region_id + '.bad.pickle',
                          bad_results2)

                # Quick summary plots
                t_sse_distribution = np.asarray([z[1] for z in good_results2])

                pvalue = np.sum(t_sse_distribution >= t_sse_data) / (1.0 * nsample)
                print('Posterior predictive p-value = %f' % (pvalue))

                plt.figure(2)
                plt.hist(t_sse_distribution, bins=30)
                plt.xlabel('SSE statistic')
                plt.ylabel('Number found [%i samples]' % (nsample))
                plt.title(title)
                plt.axvline(t_sse_data, color='k', label='p=%4.3f [$T_{SSE}=$%f]' % (pvalue, t_sse_data))
                plt.legend(fontsize=8)
                plt.savefig(savefig + obstype + '.posterior_predictive.T_SSE.png')
                plt.close('all')
