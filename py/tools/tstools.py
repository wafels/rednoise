#
# Tools to work with time series objects
#

import numpy as np
from astropy import units as u

from tools.timeseries import TimeSeries


# Normalized dimensionless times
def normalized_times(times, norm=None):
    if norm is None:
        normalization = times.dt.value
    else:
        normalization = norm.value
    return Timeseries.SampleTimes(times.t.value / normalization * u.dimensionless_unscaled)


# Get cadences
def cadences(t):
    return t[1:] - t[0:-1]


# Find segments of the data that have approximately equal cadences
def segment_indices(times, absolute_tolerance=0.5):
    """
    Find segments in the data where segments of the sample times have
    cadences below the specified absolute tolerance.

    Parameters
    ----------



    Returns
    -------
    indices : list
        A list of lists.  Each list is the start and end index of
        the sample times that indicates the start and end of a
        segment of the sample times where the cadences are below
        the specified absolute tolerance.
    """
    # raw cadences
    c = cadences(times)
    n = len(c)
    segments = []
    istart = 0
    iend = 1
    while iend <= n - 2:
        c0 = c[istart]
        c1 = c[iend]
        while (np.abs(c1 - c0) < absolute_tolerance) and (iend <= n - 2):
            iend = iend + 1
            c0 = c[istart]
            c1 = c[iend]
        segments.append([istart, iend])
        istart = iend
    return segments


def longest_evenly_sampled(t, absolute_tolerance=0.5):
    """Find which cadence segments are the longest

    Parameters
    ----------



    Returns
    -------

    """
    segments = segment_indices(t, absolute_tolerance=absolute_tolerance)
    segment_lengths = [seg[1] - seg[0] for seg in segments]
    which_segments = (np.max(segment_lengths) == segment_lengths).nonzero()[0]
    return [segments[k] for k in which_segments]


def is_evenly_sampled(t, absolute_tolerance):
    """

    :param t:
    :param absoluteTolerance:
    :return:
    """
    if len(segment_indices(t, absolute_tolerance=absolute_tolerance)) == 1:
        return True
    else:
        return False

#
# Frequency tools
#
# Normalized dimensionless frequencies
def normalized_freqs(f, norm=None):
    if norm is None:
        normalization = np.min(f.pf.value)
    else:
        normalization = norm.value
    return Timeseries.Frequencies(f.value / normalization * u.dimensionless_unscaled)
