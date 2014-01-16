# Test 6: Posterior predictive checking
import os
from matplotlib import pyplot as plt
from cubetools import get_datacube
from timeseries import TimeSeries
import pickle
import numpy as np
import cubetools
plt.ion()


def rn4(location, derotate=False, Xrange=None, Yrange=None):
    # Main directory where the data is
    directory = os.path.expanduser(location)
    # Which wavelength to look at

    # Load in the data using a very specific piece of code that will cut down
    # the region quite dramatically
    print('Loading data from ' + directory)

    dc, maps = get_datacube(directory, derotate=derotate)
    # Get some properties of the datacube
    ny = dc.shape[0]
    nx = dc.shape[1]
    if Xrange is None:
        xr = [0, nx - 1]
        #xrname = ''
    else:
        xr = Xrange
        #xrname = 'x' + str(xr[0]) + '_' + str(xr[1])
    if Yrange is None:
        yr = [0, ny - 1]
        #yrname = ''
    else:
        yr = Yrange
        #yrname = 'y' + str(yr[0]) + '_' + str(yr[1])

    return dc[yr[0]:yr[1], xr[0]:xr[1], :], maps


def get_pixel_locations(iput=None, nsample=100):
    """
    Define a set of pixel locations or load them in from file.
    """
    # Load the locations, or create them.
    if isinstance(iput, (str, unicode)):
        pixel_locations = pickle.load(open(iput, 'rb'))
    else:
        pixel_locations = zip(np.random.randint(0, high=iput[0], size=nsample),
                              np.random.randint(0, high=iput[1], size=nsample))
         
    return pixel_locations


def get_tslist(dc, pixel_locations, name=''):
    """
    Define time-series from the datacube.
    """
    # Get some properties of the datacube
    nt = dc.shape[2]
    
    # Create the sample time array
    dt = 12.0
    t = dt * np.arange(0, nt)
    
    # Define a list of time-series
    tslist = []
    for pxy in pixel_locations:
        ts = TimeSeries(t, dc[pxy[0], pxy[1]])
        ts.name = name + '(' + str(pxy[0]) + ',' + str(pxy[1]) + ')'
        tslist.append(ts)

    return tslist

def location_branch(location_root, branches):
    """Recursively adds a branch to a directory listing"""
    loc = os.path.expanduser(location_root)
    for branch in branches:
        loc = os.path.join(loc, branch)
    return loc


def save_location_calculator(roots, branches):
    """Takes a bunch of roots and creates subdirectories as needed"""
    locations = {}
    for k in roots.keys():
        loc = location_branch(roots[k], branches)
        cubetools.makedirs(loc)
        locations[k] = loc

    return locations


def ident_creator(branches):
    return '_'.join(branches)


# Plot an image of the regions
def plot_square(x, y, **kwargs):
    plt.plot([x[0], x[0]], [y[0], y[1]], **kwargs)
    plt.plot([x[0], x[1]], [y[1], y[1]], **kwargs)
    plt.plot([x[1], x[1]], [y[0], y[1]], **kwargs)
    plt.plot([x[0], x[1]], [y[0], y[0]], **kwargs)