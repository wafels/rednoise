#
# Some magic numbers and parameters used in various analysis programs
#
import numpy as np
import lnlike_model_fit

# Reduced chi-squared
rchi2s = '$\chi^{2}_{r}$'
rchi2string = '$<$' + rchi2s + '$<$'

# Reduced chi-squared limit colors
rchi2limitcolor = ['r', 'y']

# Look at those results that have chi-squared values that give rise to
# probabilities within these values
pvalue = np.array([0.025, 0.975])

# Other parameter names
othernames = ['total emission', 'maximum emission', 'minimum emission',
              'standard deviation of the emission',
              'standard deviation of log(emission)', 'rchi2']

# Fitting details
fitdetails = ['rchi2']

# Where in the main output from step3 is everything
ic = {"AIC": [3], "BIC": [4]}
rchi2 = [2]
success = [1, 'success']

# Model fit parameter names
def parameters(model_names):
    result = {}
    if 'power law with constant' in model_names:
        result['power law with constant'] = "amplitude", "power law index", "background"

    if 'power law with constant and lognormal' in model_names:
        result['power law with constant and lognormal'] = "amplitude", "power law index", "background", "lognormal amplitude", "lognormal position", "lognormal width"

    return result


# Number of parameters
def nparameters(model_names):
    result = {}
    if 'power law with constant' in model_names:
        result['power law with constant'] = 3

    if 'power law with constant and lognormal' in model_names:
        result['power law with constant and lognormal'] = 6

    return result


# Are the above parameters comparable to values found in ireland et al 2015?
def comparable(model_names):
    result = {}
    if 'power law with constant' in model_names:
        result['power law with constant'] =False, True, False

    if 'power law with constant and lognormal' in model_names:
        result['power law with constant and lognormal'] = False, True, False, False, False, False

    return result


# Conversion factors to convert the stored parameter values to ones which are
# simpler to understand when plotting them out
def conversion(model_names):
    return {"amplitude": 1.0 / np.log(10.0),
            "power law index": 1,
            "background": 1.0 / np.log(10.0),
            "lognormal amplitude": 1.0 / np.log(10.0),
            "lognormal position": 1.0 / np.log(10.0),
            "lognormal width": 1.0 / np.log(10.0),
            'total emission': 1.0,
            'maximum emission':  1.0,
            'minimum emission':  1.0,
            'standard deviation of the emission': 1.0,
            'standard deviation of log(emission)': 1.0,
            'rchi2': 1.0}


# Informative plot labels
def plotname(model_names):
    return {"amplitude": '$\log_{10}(A)$',
            "power law index": "power law index",
            "background": "$\log_{10}(C)$",
            "lognormal amplitude": "$\log_{10}(A_{G})$",
            "lognormal position": "$\log_{10}(f_{G})$",
            "lognormal width": "$\log_{10}(\sigma_{G})$",
            'total emission': 'sum(E)',
            'maximum emission':  'max(E)',
            'minimum emission':  'min(E)',
            'standard deviation of the emission': 'sd(E)',
            'standard deviation of log(emission)': 'sd(log(E))',
            'rchi2': rchi2s}

#
# Calculate reduced chi-squared limits given pvalues
#
def rchi2limit(pvalue, nposfreq, nparameters):
    x = {}
    for model_name in nparameters.keys():
        n = nparameters[model_name]
        x[model_name] = [lnlike_model_fit.rchi2_given_prob(pvalue[1], 1.0, nposfreq - n - 1),
                         lnlike_model_fit.rchi2_given_prob(pvalue[0], 1.0, nposfreq - n - 1)]
    return x


# Create a label which shows the limits of the reduced chi-squared value. Also
# define some colors that signify the upper and lower levels of the reduced
# chi-squared/
def rchi2label(rchi2limit):
    return '%1.2f%s%1.2f' % (rchi2limit[0], rchi2string, rchi2limit[1])

#
# Probability string that defines a percentage output
#
def percentstring(pvalue):
    return '%2.1f%%<p<%2.1f%%' % (100 * pvalue[0], 100 * pvalue[1])


def percent_lo(pvalue):
    return '%s (p$<$%2.1f%%)' % (rchi2s, 100 * pvalue[0])


def percent_hi(pvalue):
    return '%s (p$>$%2.1f%%)' % (rchi2s, 100 - 100 * pvalue[1])

#
# Create a mask that shows where there was a successful fit, and the reduced
# chi-squared is inside the required limits.  True equals a value we want to
# keep, false equals a value we don't want.
#
def result_filter(success, rchi2, rchilimit):
    rchi2_gt_low_limit = rchi2 > rchilimit[0]
    rchi2_lt_high_limit = rchi2 < rchilimit[1]
    return success * rchi2_gt_low_limit * rchi2_lt_high_limit


def summary_statistics(a, bins=100):
    hist, bin_edges = np.histogram(a, bins)
    hist_max_index = np.argmax(hist)
    mode = 0.5*(bin_edges[hist_max_index] + bin_edges[hist_max_index+1])
    return {"mean": np.mean(a),
            "median": np.percentile(a, 50.0),
            "lo": np.percentile(a, 2.5),
            "hi": np.percentile(a, 97.5),
            "min":np.min(a),
            "max": np.max(a),
            "std": np.std(a),
            "mode": mode}

#
# Periods are the inverse of frequencies.  Frequencies are returned as a
# normalized value in base 10, and normalized.  If the normalization factor is
# f_norm, and the original frequency is f, then what is returned in the Fit
# object is
#
# log_{10}(f/f_norm)
#
# The function below returns the corresponding period for normalized
# input frequency nu
def convert_to_period(f_norm, nu):
    return 1.0 / (f_norm * 10.0 ** nu)
