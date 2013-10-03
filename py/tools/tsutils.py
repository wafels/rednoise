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


def cadence(t, absoluteTolerance=0.5):
    """Get some information on the observed cadences"""
    # raw cadences
    cadences = t[1:] - t[0:-1]
    n = len(cadences)
    segments = []
    iStart = 0
    iEnd = 1
    while iEnd <= n - 2:
        c0 = cadences[iStart]
        c1 = cadences[iEnd]
        while (np.abs(c1 - c0) < absoluteTolerance) and (iEnd <= n - 2):
            iEnd = iEnd + 1
            c0 = cadences[iStart]
            c1 = cadences[iEnd]
        segments.append([iStart, iEnd])
        iStart = iEnd
    return segments


def longest_evenly_sampled(t, absoluteTolerance=0.5):
    """Find which segments are the longest"""
    segments = cadence(t, absoluteTolerance=absoluteTolerance)
    segment_lengths = [seg[1] - seg[0] for seg in segments]
    which_segments = (np.max(segment_lengths) == segment_lengths).nonzero()[0]
    return [segments[k] for k in which_segments]
