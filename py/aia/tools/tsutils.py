"""
Helper routines for time-series
"""

import numpy as np


def autocorrelate(data):
    """
    Calculate the autocorrelation function of some input data.  Copied more or
    less directly from http://stackoverflow.com/questions/643699/
    how-can-i-use-numpy-correlate-to-do-autocorrelation
    """
    yunbiased = data - np.mean(data)
    ynorm = np.sum(yunbiased**2)
    acor = np.correlate(yunbiased, yunbiased, "same") / ynorm
    # use only second half
    return acor[len(acor)/2:]


def fix_nonfinite(data):
    """
    Finds all the nonfinite regions in the data and replaces them with a simple
    linear interpolation.
    """
    good_indexes = np.isfinite(data)
    bad_indexes = np.logical_not(good_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data


def unique_spatial_subsamples(shape, n):
    """ Given a particular input array shape, generate 'n' unique random
    locations"""
    isunique = False
    while isunique is False:
        rand_1 = np.random.randint(0, high=shape[0], size=n)
        rand_2 = np.random.randint(0, high=shape[1], size=n)
        locations = zip(rand_1, rand_2)
        if len(locations) == len(list(set(locations))):
            isunique = True
    return locations


def frequency_response_function(index, weight, angularfreq):
    w = np.zeros_like(angularfreq, dtype=np.complex64)
    for i, f in enumerate(angularfreq):
        onent = np.zeros_like(index) + np.complex(0, -f) * index
        w[i] = np.sum(weight * np.exp(onent))
    return w


def transfer_function(index, weight, angularfreq):
    w = frequency_response_function(index, weight, angularfreq)
    return np.abs(w) ** 2


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')
