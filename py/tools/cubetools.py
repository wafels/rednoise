import numpy as np
import sunpy
import matplotlib.pyplot as plt
from scipy.io import readsav
import os
from sunpy.coords.util import rot_hpc
import tsutils
import pickle

# For image co-registration
from skimage.feature import match_template

# Shift an image by a given amount - subpixel shifts are permitted
from scipy.ndimage.interpolation import shift

# Used in the fitting of Gaussians


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

    # maximum differences found so far
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

        # In matplotlib, the origin of the image is in the top right of the
        # the array.  Therefore, shifting in the y direction has to be reversed
        # Also, the first dimension in the numpy array refers to what a solar
        # physicist refers to as the north-south (i.e., 'y') direction.
        if np.abs(ydiff) >= diff_limit and np.abs(xdiff) >= diff_limit:
            shifted = shift(m.data, [-ydiff, -xdiff])
        else:
            shifted = m.data

        # Store all the shifted data.
        datacube[..., t] = shifted

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
        axes.set_title("%i" % (z))
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
# Shift a datacube according to calculated co-registration displacements
#
def coregister_datacube(template, datacube, register_index=0):
    """
    
    """
    # Number of layers
    nt = datacube.shape[2]
    
    # Calculate the displacements of each layer in the datacube relative to the
    # known position of a template
    x_displacement, y_displacement = calculate_coregistration_displacements(template, datacube)

    # Shift each layer of the datacube the required amounts
    for i in range(0, nt-1):
        layer = datacube[:, :, i]
        y_diff = y_displacement[i] - y_displacement[register_index]
        x_diff = x_displacement[i] - x_displacement[register_index]
        shifted = shift(layer, [-y_diff, -x_diff])
        datacube[:, :, i] = shifted

    return datacube, x_displacement - x_displacement[register_index], y_displacement - y_displacement[register_index]


#
# Calculate the co-registration displacements for a datacube
#
def calculate_coregistration_displacements(template, datacube):
    """
    Calculate the coregistration of (ny, nx) layers in a (ny, nx, nt) datacube
    against a chosen template.  All inputs are assumed to be numpy arrays.
    
    Inputs
    ------
    template : a numpy array of size (N, M) where N < ny and M < nx .

    datacube : a numpy array of size (ny, nx, nt), where the first two
               dimensions are spatial dimensions, and the third dimension is
               time.

    Outputs
    -------
    
    
    """
    nt = datacube.shape[2]

    # Storage for the results
    keep_x = []
    keep_y = []

    # Go through each layer and perform the matching
    for t in range(0, nt):
        # Get the layer
        layer = datacube[:, :, t]
    
        # Match the template to the layer
        result = match_template(layer, template)
    
        # Get the index of the maximum in the correlation function
        ij = np.unravel_index(np.argmax(result), result.shape)
        cor_max_x, cor_max_y = ij[::-1]

        # Get the correlation function around the maximum
        array_around_maximum = result[np.max([0, cor_max_y - 1]): np.min([cor_max_y + 2, result.shape[0] - 1]), 
                                      np.max([0, cor_max_x - 1]): np.min([cor_max_x + 2, result.shape[1] - 1])]
        y_shift_relative_to_maximum, x_shift_relative_to_maximum = \
        get_correlation_shifts(array_around_maximum)
    
        # Get shift relative to correlation array
        y_shift_relative_to_correlation_array = y_shift_relative_to_maximum + cor_max_y
        x_shift_relative_to_correlation_array = x_shift_relative_to_maximum + cor_max_x
        
        # Store the results
        keep_x.append(x_shift_relative_to_correlation_array)
        keep_y.append(y_shift_relative_to_correlation_array)

    return np.asarray(keep_x), np.asarray(keep_y)


def get_correlation_shifts(array):
    """
    Estimate the location of the maximum of a fit to the input array.  The
    estimation in the x and y directions are done separately. The location
    estimates can be used to implement subpixel shifts between two different
    images.

    Inputs
    ------
    array : an array with at least one dimension that has three elements.  The
            input array is at most a 3 x 3 array of correlation values
            calculated by matching a template to an image.

    Outputs
    -------
    y, x : the location of the peak of a parabolic fit.
    
    """
    # Check input shape
    ny = array.shape[0]
    nx = array.shape[1]
    if nx > 3 or ny > 3:
        print 'Input array is too big in at least one dimension. Returning Nones'
        return None, None

    # Find where the maximum of the input array is
    ij = np.unravel_index(np.argmax(array), array.shape)
    x_max_location, y_max_location = ij[::-1]

    # Estimate the location of the parabolic peak if there is enough data.
    # Otherwise, just return the location of the maximum in a particular
    # direction.
    if ny == 3:
        y_location = parabolic_turning_point(array[:, x_max_location])
    else:
        y_location = 1.0 * y_max_location
    
    if nx == 3:
        x_location = parabolic_turning_point(array[y_max_location, :])
    else:
        x_location = 1.0 * x_max_location
    
    return y_location, x_location


def parabolic_turning_point(y):
    """
    Find the location of the turning point for a parabola f(x) = ax^2 + bx + c
    The maximum is located at x0 = -b / 2a .  Assumes that the input array
    represents an equally spaced sampling at the locations f(-1), f(0) and f(1)
    """
    numerator = -0.5 * y.dot([-1, 0, 1])
    denominator = y.dot([1, -2, 1])
    return numerator / denominator
