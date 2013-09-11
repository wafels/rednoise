"""
Helper routines for time-series
"""

import numpy as np


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
    """ Given a particular input array shape, generate 'n' random locations"""
    # Unique random locations
    isunique = False
    while isunique is False:
        rand_1 = np.random.randint(0, high=shape[0], size=n)
        rand_2 = np.random.randint(0, high=shape[1], size=n)
        locations = zip(rand_1, rand_2)
        if len(locations) == len(list(set(locations))):
            isunique = True
    return locations


def cadence(t, absoluteTolerance=0.5):
    """#Get some information on the observed cadences"""
    """
    cadences = t.time[1:] - t.time[0:-1]
    segments = []
    iStart = 0
    iEnd = 0
    nsegment = 0
    repeat begin
        repeat begin
            iEnd = iEnd + 1
            c0 = cadence[iStart]
            c1 = cadence[iEnd]
        endrep until (abs(c1-c0) gt absoluteTolerance) or (iEnd eq nt-2)
        segment.append([iStart, iEnd])
        nsegment = nsegment + 1
        iStart = iEnd
    endrep until (iEnd eq nt-2)
"""
