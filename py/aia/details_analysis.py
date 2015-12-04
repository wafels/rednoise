#
# Some magic numbers and parameters used in various analysis programs
#
from copy import deepcopy
import numpy as np
import astropy
import astropy.units as u
import lnlike_model_fit


# Information criterion limit
# For the BIC, see https://en.wikipedia.org/wiki/Bayesian_information_criterion
# and Kass, Robert E.; Raftery, Adrian E. (1995), "Bayes Factors", Journal of
# Setting the ic_limit to 6.0 for the BIC picks out models that have "strong"
# evidence in their favour according to the Kass and Raftery 1995 above.

# Information criteria we want to examine
ic_details = {'BIC': [6.0, 0.0]}


# Where in the main output from step3 is everything
ic = {"AIC": [3], "BIC": [4]}
rchi2 = [2]
success = [1, 'success']


#
# When looking at the data, we want to remove outliers.  The limits below
# set thresholds on the fit parameters and derived values
#

limits = {"standard": {"power law index": [0., 7.] * u.dimensionless_unscaled,
                       "ln(constant)": [0., 10.] * u.dimensionless_unscaled,
                       "ln(power law amplitude)": [0.0, 10.0] * u.dimensionless_unscaled,
                       "ln(lognormal width)": [0.0, 0.6] * u.dimensionless_unscaled,
                       "lognormal position": [1.0/(1800 * 12.0), 0.015] * u.Hz,
                       "ln(lognormal amplitude)": [0.0, 10.0] * u.dimensionless_unscaled,
                       "period": [24.0, 3000.0] * u.s,
                       "frequency": [1.0/(1800 * 12.0), 1.0/(2 * 12.0)] * u.Hz,
                       "ratio": [-5.0, 5.0] * u.dimensionless_unscaled,
                       "power law index below break": [0., 7.] * u.dimensionless_unscaled,
                       "power law index above break": [0., 7.] * u.dimensionless_unscaled,
                       "dimensionless frequency": [1.0/(1800 * 12.0), 0.015] * u.Hz}}

limits["low_lognormal_width"] = deepcopy(limits["standard"])
limits["low_lognormal_width"]["ln(lognormal width)"] = [0.0, 0.1] * u.dimensionless_unscaled

limits["low_lognormal_width_3_to_5_minutes"] = deepcopy(limits["low_lognormal_width"])
limits["low_lognormal_width_3_to_5_minutes"]["lognormal position"] = [1.0/500.0, 1.0/160.0] * u.Hz

limits["high_lognormal_width"] = deepcopy(limits["standard"])
limits["high_lognormal_width"]["ln(lognormal width)"] = [0.1, 0.6] * u.dimensionless_unscaled

limits["high_lognormal_width_freq_less_than_5_minutes"] = deepcopy(limits["standard"])
limits["high_lognormal_width_freq_less_than_5_minutes"]["lognormal position"] = [1.0/(1800.0 * 12.0), 1.0/501.0] * u.Hz


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


def summary_statistics(a, bins=100):
    hist, bin_edges = np.histogram(a, bins)
    hist_max_index = np.argmax(hist)
    mode = 0.5*(bin_edges[hist_max_index] + bin_edges[hist_max_index+1])
    if isinstance(a, astropy.units.Quantity):
        unit = a.unit
    else:
        unit = 1.0
    return {"mean": np.mean(a),
            "median": np.percentile(a, 50.0) * unit,
            "lo": np.percentile(a, 2.5) * unit,
            "hi": np.percentile(a, 97.5) * unit,
            "min": np.min(a),
            "max": np.max(a),
            "std": np.std(a),
            "mode": mode * unit}


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


#
# Get the mode of a histogram
#
def get_mode(h_info):
    i = np.argmax(h_info[0])
    bin_edges = h_info[1][i: i+2]
    return h_info[0][i], bin_edges


#
# Define a useful text string describing the number of masked entries.  It is
# assumed that the input mask is a numpy mask where True means masked
#
def get_mask_info(mask):
    n_not_masked = np.sum(np.logical_not(mask))
    return '(#px=%i, used=%3.1f%%)' % (n_not_masked, 100 * n_not_masked/np.float64(mask.size))


# Define a mask where the IC say that one model is preferred over another
def get_ic_location(this, this_model, model_names, ic_type=None, ic_limits=None):
    # Get the data for the first model
    this_model_data = this[this_model]

    # Get the data for the other model
    other_model = model_names[1 - model_names.index(this_model)]
    other_model_data = this[other_model]

    # Where the information criteria are less than zero indicates the
    # preferred model
    dAIC = this_model_data.as_array("AIC") - other_model_data.as_array("AIC")
    dBIC = this_model_data.as_array("BIC") - other_model_data.as_array("BIC")

    # Return locations where either dAIC > 0, dBIC > 0 or both
    if ic_type == 'none':
        mask = np.ones_like(dAIC, dtype=bool)

    if ic_limits is None:
        ic_limit = 0.0
    else:
        ic_limit = ic_limits[ic_type]

    if ic_type == 'AIC':
        mask = dAIC < ic_limit

    if ic_type == 'BIC':
        mask = dBIC < ic_limit

    if ic_type == 'both':
        mask_aic = dAIC < ic_limit
        mask_bic = dBIC < ic_limit
        mask = mask_aic * mask_bic

    # Return the indices of where to mask (assuming that the mask is to be used
    # in a numpy masked array.
    return np.where(np.logical_not(mask))

sqrt_two_pi = np.sqrt(2 * np.pi)
def integrate_lognormal_parameters(amplitude, width):
    return sqrt_two_pi * width * 10.0 ** amplitude