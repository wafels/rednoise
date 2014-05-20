"""
Helpers for the paper
"""
from copy import deepcopy
import os
import csv
import cPickle as pickle
import numpy as np


# Prints a title, with space before it and an underline
def prettyprint(s, underline='-'):
    print(' ')
    print(s)
    print(str(underline) * len(s))


def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation.
    The algorithm plots out nicely formatted explicit numbers for values
    greater and less then 1.0."""
    if x >= 1.0:
        return '%i' % (x)
    else:
        n = 0
        while x * (10 ** n) <= 1:
            n = n + 1
        fmt = '%.' + str(n - 1) + 'f'
        return fmt % (x)


def power_distribution_details():
    return {"xlim": [-8.0, 1.5], "ylim": [0.0, 0.12]}


def fit_details():
    return {"ylim": [0.00001, 10.0]}


#
# Line styles for all figures
#
class LineStyles:
    def __init__(self):
        self.color = 'b'
        self.linewidth = 1
        self.linestyle = "-"
        self.label = ""

s171 = LineStyles()
s171.color = 'b'

s_U68 = deepcopy(s171)
s_U68.label = '68%, upper'
s_U68.linestyle = "-"

s_L68 = deepcopy(s_U68)
s_L68.label = '68%, lower'
s_L68.color = 'r'

s_U95 = deepcopy(s171)
s_U95.label = '95%, upper'
s_U95.linestyle = "--"

s_L95 = deepcopy(s_U95)
s_L95.label = '95%, lower'
s_L95.color = 'r'


s193 = LineStyles()
s193.color = 'r'


s5min = LineStyles()
s5min.color = 'k'
s5min.linewidth = 1
s5min.linestyle = "-"
s5min.label = '5 mins.'

s3min = LineStyles()
s3min.color = 'k'
s3min.linewidth = 1
s3min.linestyle = "--"
s3min.label = '3 mins.'


#
# String defining the basics number of time series
#
def tsDetails(nx, ny, nt):
    return '[%i t.s., %i samples]' % (nx * ny, nt)

#
# Nicer names for printing
#
sunday_name = {'moss': 'moss',
               'qs': 'quiet Sun',
               'loopfootpoints': 'loop footpoints',
               'loopfootpoints1': 'loop footpoints 1',
               'loopfootpoints2': 'loop footpoints 2',
               'highlimb': 'high limb',
               'lowlimb': 'low limb',
               'crosslimb': 'cross limb',
               'sunspot': 'sunspot'}


# Write a CSV time series
def csv_timeseries_write(location, fname, a):
    # Get the time and the data out separately
    t = a[0]
    d = a[1]
    # Make the directory if it is not already
    if not(os.path.isdir(location)):
        os.makedirs(location)
    # full file name
    savecsv = os.path.join(location, fname)
    # Open and write the file
    ofile = open(savecsv, "wb")
    writer = csv.writer(ofile, delimiter=',')
    for i, dd in enumerate(d):
        writer.writerow([t[i], dd])
    ofile.close()


# Write out a pickle file
def pkl_write(location, fname, a):
    pkl_file_location = os.path.join(location, fname)
    print('Writing ' + pkl_file_location)
    pkl_file = open(pkl_file_location, 'wb')
    for element in a:
        pickle.dump(element, pkl_file)
    pkl_file.close()


def fix_nonfinite(data):
    bad_indexes = np.logical_not(np.isfinite(data))
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

