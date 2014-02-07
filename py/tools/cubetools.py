import numpy as np
import scipy
import sunpy
import matplotlib.pyplot as plt
from scipy.io import readsav
import os
from sunpy.coords.util import rot_hpc
import tsutils
import pickle

# For image co-registration
from skimage.feature import match_template


def derotated_datacube_from_mapcube(maps, ref_index=0, diff_limit=0.01):
    """Return a datacube from a set of maps"""

    # get the dimensions of the datacube
    nt = len(maps[:])
    ny = maps[0].shape[0]
    nx = maps[0].shape[1]

    # reference values
    ref_center = maps[ref_index].center
    ref_time = maps[ref_index].date

    # Output datacube
    datacube = np.zeros((ny, nx, nt))

    # maximum differences found si far
    max_abs_xdiff = 0.0
    max_abs_ydiff = 0.0

    # Assume all the maps have the same pointing and the Sun rotates in the
    # field of view.  The center of the field of view at the start time gives
    # the initial field of view that then moves at later times.  This initial
    # location has to be rotated forward to get the correct center of the FOV
    # for the next maps.
    for t, m in enumerate(maps):
        newx, newy = rot_hpc(ref_center['x'], ref_center['y'], ref_time,
                                          m.date)
        if newx is None:
            xdiff = 0
        else:
            xdiff = (newx - ref_center['x']) / m.scale['x']
        if newy is None:
            ydiff = 0
        else:
            ydiff = (newy - ref_center['y']) / m.scale['y']

        if np.abs(xdiff) > max_abs_xdiff:
            max_abs_xdiff = np.ceil(np.abs(xdiff))

        if np.abs(ydiff) > max_abs_ydiff:
            max_abs_ydiff = np.ceil(np.abs(ydiff))

        #print xdiff, ydiff
        # In matplotlib, the origin of the image is in the top right of the
        # the array.  Therefore, shifting in the y direction has to be reversed
        # Also, the first dimension in the numpy array refers to what a solar
        # physicist refers to as the north-south (i.e., 'y') direction.
        #print t, xdiff, ydiff
        if np.abs(ydiff) >= diff_limit and np.abs(xdiff) >= diff_limit:
            shifted = scipy.ndimage.interpolation.shift(m.data, [-ydiff, -xdiff])
        else:
            shifted = m.data
        # Store all the shifted data.
        datacube[..., t] = shifted
        #print t, m.date, maps[t].center, xdiff, ydiff

    # Zero out the edges where the data has been moved
    return datacube[0:ny - max_abs_ydiff - 1, 0:nx - max_abs_xdiff - 1:, :]


def visualize(datacube, delay=0.1):
    """
    Visualizes a datacube.  Requires matplotlib 1.1
    """
    nt = datacube.shape[2]
    print nt
    fig = plt.figure()

    axes = fig.add_subplot(111)
    axes.set_xlabel('X-position')
    axes.set_ylabel('Y-position')

    #extent = wave_maps[0].xrange + wave_maps[0].yrange
    #axes.set_title("%s %s" % (wave_maps[0].name, wave_maps[0].date))

    img = axes.imshow(datacube[:, :, 0], origin='lower')
    #fig.colorbar(img)
    fig.show()

    for z in np.arange(1, nt - 1):
        m = datacube[:, :, z]
        #axes.set_title("%i" % (i))
        img.set_data(m)
        plt.pause(delay)
    return None


def get_datacube(path, derotate=False, correlate=False):
    """
    Function that goes to a directory and returns a datacube
    """
    if os.path.isfile(path):
        idl = readsav(path)
        return np.swapaxes(np.swapaxes(idl['region_window'], 0, 2), 0, 1)
    else:
        # Get a mapcube
        maps = sunpy.Map(path, cube=True)
        if derotate:
            dc = derotated_datacube_from_mapcube(maps)
        else:
            nt = len(maps[:])
            ny = maps[0].shape[0]
            nx = maps[0].shape[1]
            dc = np.zeros((ny, nx, nt))
            for i, m in enumerate(maps):
                dc[:, :, i] = m.data[:, :]
        if correlate:
            dc = coregister_datacube(dc)
        return dc, maps


def coregister_datacube(dc, register_index=0, diff_limit=0.01):
    """
    Co-register a series of layers in a datacube
    """
    # data cube shape
    ny = dc.shape[0]
    nx = dc.shape[1]
    nt = dc.shape[2]

    # Define a template at the registration layer
    center = [ny / 2, nx / 2]
    template_y = [center[0] - ny / 4, center[0] + ny / 4]
    template_x = [center[1] - nx / 4, center[1] + nx / 4]
    template = dc[template_y[0]:template_y[1],
                  template_x[0]:template_x[1],
                  register_index]

    # Go through each layer and perform the matching
    for t in range(0, nt):
        # Get the layer
        layer = dc[:, :, t]
        # Get the cross correlation function
        result = match_template(layer, template)

        # Fit a 2-dimensional Gaussian to the correlation peak and get the
        # displacement
        gaussian_parameters = fitgaussian(result)
        ydiff = gaussian_parameters[2] - center[0]
        xdiff = gaussian_parameters[1] - center[1]

        # Shift the layer
        if np.abs(ydiff) >= diff_limit and np.abs(xdiff) >= diff_limit:
            shifted = scipy.ndimage.interpolation.shift(layer, [-ydiff, -xdiff])
        else:
            shifted = layer
        dc[..., t] = shifted
    return dc


def sum_over_space(dc, remove_mean=False):
    """
    Sum a datacube over all the spatial locations
    """
    # Get some properties of the datacube
    ny = dc.shape[0]
    nx = dc.shape[1]
    nt = dc.shape[2]

    # Storage for the
    fd = np.zeros((nt))
    for i in range(0, nx):
        for j in range(0, ny):
            d = dc[j, i, :].flatten()

            # Fix the data for any non-finite entries
            d = tsutils.fix_nonfinite(d)

            # Remove the mean
            if remove_mean:
                d = d - np.mean(d)

            # Sum up all the data
            fd = fd + d
    return fd


# Data that we want to save
def save_output(filename, datacube, times, pixel_index):
    outputfile = open(filename, 'wb')
    pickle.dump(datacube, outputfile)
    pickle.dump(times, outputfile)
    pickle.dump(pixel_index, outputfile)
    outputfile.close()
    print('Saved to ' + filename)
    return


# Save multiple regions
def save_region(dc, output, regions, wave, times):
    keys = regions.keys()
    for region in keys:
        pixel_index = regions[region]
        y = pixel_index[0]
        x = pixel_index[1]
        #filename = region + '.' + wave + '.' + str(pixel_index) + '.datacube.pickle'
        filename = region + '.' + wave + '.datacube.pickle'
        save_output(os.path.join(output, filename),
                    dc[y[0]: y[1], x[0]:x[1], :],
                    times,
                    pixel_index)


def makedirs(output):
    if not os.path.isdir(output):
        os.makedirs(output)


#
# 2d Gaussian fitting code
#
# Taken from http://wiki.scipy.org/Cookbook/FittingData
#
#
def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = np.float(width_x)
    width_y = np.float(width_y)
    return lambda x, y: height * np.exp(
                                  -(((center_x - x) / width_x) ** 2
                                    + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = scipy.optimize.leastsq(errorfunction, params)
return p
