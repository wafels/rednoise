#
# Tools to work with time series objects
#

import numpy as np
from astropy import units as u

from tools.timeseries import SampleTimes, Frequencies


# Normalized dimensionless times
def normalized_times(times, norm=None):
    if norm is None:
        normalization = times.dt.value
    else:
        normalization = norm.value
    return SampleTimes(times.t.value / normalization * u.dimensionless_unscaled)


# Get cadences
def cadences(times):
    return times.t[1:] - times.t[0:-1]


# Find segments of the data that have approximately equal cadences
def segment_indices(times, absolutetolerance=0.5):
    """
    Find segments in the data where segments of the sample times have
    cadences below the specified absolute tolerance.

    Returns
    -------
    indices : list
        A list of lists.  Each list is the start and end index of
        the sample times that indicates the start and end of a
        segment of the sample times where the cadences are below
        the specified absolute tolerance.
    """
    # raw cadences
    n = len(cadences(times))
    segments = []
    istart = 0
    iend = 1
    while iend <= n - 2:
        c0 = cadences[istart]
        c1 = cadences[iend]
        while (np.abs(c1 - c0) < absolutetolerance) and (iend <= n - 2):
            iend = iend + 1
            c0 = cadences[istart]
            c1 = cadences[iend]
        segments.append([istart, iend])
        istart = iend
    return segments


#
# Frequency tools
#
# Normalized dimensionless frequencies
def normalized_freqs(f, norm=None):
    if norm is None:
        normalization = np.min(f.pf.value)
    else:
        normalization = norm.value
    return Frequencies(f.value / normalization * u.dimensionless_unscaled)
