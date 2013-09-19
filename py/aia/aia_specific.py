# Test 6: Posterior predictive checking
import os
from matplotlib import pyplot as plt
from cubetools import get_datacube
plt.ion()


def rn4(wave, Xrange=None, Yrange=None):
    # Main directory where the data is
    location = '~/Data/AIA_Data/rn4'
    maindir = os.path.expanduser(location)
    # Which wavelength to look at

    # Construct the directory
    directory = os.path.join(maindir, wave)

    # Load in the data using a very specific piece of code that will cut down
    # the region quite dramatically
    print('Loading data from ' + directory)

    dc = get_datacube(directory)
    # Get some properties of the datacube
    ny = dc.shape[0]
    nx = dc.shape[1]
    if Xrange is None:
        xr = [0, nx - 1]
    else:
        xr = Xrange
    if Yrange is None:
        yr = [0, ny - 1]
    else:
        yr = Yrange

    return dc[yr[0]:yr[1], xr[0]:xr[1], :], directory
