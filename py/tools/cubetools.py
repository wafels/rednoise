import numpy as np
import sunpy
import matplotlib.pyplot as plt
from scipy.io import readsav
import os
from sunpy.coords.util import rot_hpc
from sunpy.map import Map
from copy import deepcopy
import tsutils
import pickle 

# For faster image co-registration
from skimage.feature import match_template

# Shift an image by a given amount - subpixel shifts are permitted
from scipy.ndimage.interpolation import shift


#
# From a directory full of FITS files, return a datacube and a mapcube
#
def get_datacube(path, derotate=False, clip=False):
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
            dc, ysrdisp, xsrdisp = derotated_datacube_from_mapcube(maps, clip=clip)
        else:
            ysrdisp = None
            xsrdisp = None
            nt = len(maps[:])
            ny = maps[0].shape[0]
            nx = maps[0].shape[1]
            dc = np.zeros((ny, nx, nt))
            for i, m in enumerate(maps):
                dc[:, :, i] = m.data[:, :]
        return dc, ysrdisp, xsrdisp, maps


def derotated_datacube_from_mapcube(maps, ref_index=0, clip=False):
    """Return a derotated datacube from a set of maps"""

    # get the dimensions of the datacube
    nt = len(maps[:])
    ny = maps[0].shape[0]
    nx = maps[0].shape[1]

    # reference values
    ref_center = maps[ref_index].center
    ref_time = maps[ref_index].date

    # Output datacube
    datacube = np.zeros((ny, nx, nt))

    # Values of the displacements
    ydiff = np.zeros(nt)
    xdiff = np.zeros(nt)

    # Assume all the maps have the same pointing and the Sun rotates in the
    # field of view.  The center of the field of view at the start time gives
    # the initial field of view that then moves at later times.  This initial
    # location has to be rotated forward to get the correct center of the FOV
    # for the next maps.
    for t, m in enumerate(maps):

        # Store all the data in a 3-d numpy array
        datacube[..., t] = m.data

        # Find the center of the field of view
        newx, newy = rot_hpc(ref_center['x'], ref_center['y'], ref_time,
                                          m.date)

        # Calculate the displacements in tems of pixels
        if newx is None:
            xdiff[t] = 0.0
        else:
            xdiff[t] = (newx - ref_center['x']) / m.scale['x']
        if newy is None:
            ydiff[t] = 0.0
        else:
            ydiff[t] = (newy - ref_center['y']) / m.scale['y']

    # shift the data cube according to the calculated displacements due to
    # solar rotation
    datacube = shift_datacube_layers(datacube, ydiff, xdiff)

    # Optionally clip the datacube to remove data that may be affected by edge
    # effects due to solar derotation.
    if clip:
        return clip_edges(datacube, ydiff, xdiff), ydiff, xdiff
    else:
        return datacube


def get_datacube_from_mapcube(mapcube):
    """
    Extract the image data from a mapcube.
    """
    # get the dimensions of the datacube
    nt = len(mapcube[:])
    ny = mapcube[0].shape[0]
    nx = mapcube[0].shape[1]

    # Output datacube
    datacube = np.zeros((ny, nx, nt))
    for t, m in enumerate(mapcube):

        # Store all the data in a 3-d numpy array
        datacube[..., t] = m.data

    # Return the datacube
    return datacube


def get_properties_from_mapcube(mapcube):
    """
    Extract the mapcube properties out of the mapcube.
    """
    pass


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

    for z in np.arange(0, nt):
        m = datacube[:, :, z]
        axes.set_title("%i" % (z))
        img.set_data(m)
        plt.pause(delay)
    return None


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


def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for
    the 1d description.  Copied from http://code.google.com/p/agpy/source/browse/trunk/agpy/psds.py?r=343
    """
    # scalar unity; don't window if dims are too small
    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        return np.hanning(N)
    else:
        return np.outer(np.hanning(M), np.hanning(N))


#
# Coalign a datacube
#
#
# Shift a datacube according to calculated co-registration displacements
#
def coregister_datacube(datacube, layer_index=0, template_index=None,
                        clip=False, func=default_map_manipulation_function):
    """
    Co-register the layers in a datacube according to a template taken from
    that datacube.

    Input
    -----
    datacube : a numpy array of shape (ny, nx, nt), where nt is the number of
               layers in the datacube.

    layer_index : the layer in the datacube from which the template will be
                  extracted.

    template_index : a array-like set of co-ordinates of the bottom left hand
                     cornor and the top right cornor of the template.  If set
                     to None, then the default template is used. The
                     template_index is defined as [ [y1, x1], [y2, x2] ].

    clip : clip off x, y edges in the datacube that are potentially affected
            by edges effects.

    func: a function which is applied to the data values before the
          coalignment method is applied.  This can be useful in coalignment,
          because it is sometimes better to co-align on a function of the data
          rather than the data itself.  The calculated shifts are applied to
          the original data.  Useful functions to consider are the log of the
          image data, or 1 / data. The function is of the form func = F(data).  
          The default function ensures that the data are floats.

    Output
    ------
    datacube : the input datacube each layer having been co-registered against
               the template.

    y_displacement : a one dimensional array of length nt with the pixel
                     y-displacements relative position of the template at the
                     value layer_index.  Note that y_displacement[layer_index]
                     is zero by definition.

    x_displacement : a one dimensional array of length nt with the pixel
                     x-displacements relative position of the template at the
                     value layer_index.  Note that x_displacement[layer_index]
                     is zero by definition.
    """

    # Calculate the template
    if template_index == None:
        ny = datacube.shape[0]
        nx = datacube.shape[1]
        template = datacube[ny / 4: 3 * ny / 4,
                            nx / 4: 3 * nx / 4,
                            layer_index]
    else:
        template = datacube[template_index[0][0]:template_index[1][0],
                            template_index[0][1]:template_index[1][1],
                            layer_index]

    # Calculate the displacements of each layer in the datacube relative to the
    # known position of a template
    y_displacement, x_displacement = calculate_coregistration_displacements(template, datacube, func)

    # Displacement relative to the layer_index
    y_displacement = y_displacement - y_displacement[layer_index]
    x_displacement = x_displacement - x_displacement[layer_index]

    # Shift each layer of the datacube the required amounts
    datacube = shift_datacube_layers(datacube, y_displacement, x_displacement)

    if clip:
        return clip_edges(datacube, y_displacement, x_displacement), y_displacement, x_displacement
    else:
        return datacube, y_displacement, x_displacement


#
# Shift layers in a datacube according to some displacements
#
def shift_datacube_layers(datacube, y_displacement, x_displacement):
    """
    Shifts the layers of a datacube by given amounts

    Input
    -----
    datacube : a numpy array of shape (ny, nx, nt), where nt is the number of
               layers in the datacube

    y_displacement : how much to shift each layer in the y - direction.  An
                     one-dimensional array of length nt.

    x_displacement : how much to shift each layer in the x - direction.  An
                     one-dimensional array of length nt.


    Output
    ------
    A numpy array of shape (ny, nx, nt).  All layers have been shifted
    according to the displacement amounts.

    """
    # Number of layers
    nt = datacube.shape[2]

    # Shift each layer of the datacube the required amounts
    for i in range(0, nt):
        layer = datacube[:, :, i]
        shifted = shift(layer, [-y_displacement[i], -x_displacement[i]])
        datacube[:, :, i] = shifted
    return datacube


#
# Calculate the co-registration displacements for a datacube
#
def calculate_coregistration_displacements(template, datacube, func):
    """
    Calculate the coregistration of (ny, nx) layers in a (ny, nx, nt) datacube
    against a chosen template.  All inputs are assumed to be numpy arrays.

    Inputs
    ------
    template : a numpy array of size (N, M) where N < ny and M < nx .

    datacube : a numpy array of size (ny, nx, nt), where the first two
               dimensions are spatial dimensions, and the third dimension is
               time.

    func: a function which is applied to the data values before the
          coalignment method is applied.  This can be useful in coalignment,
          because it is sometimes better to co-align on a function of the data
          rather than the data itself.  The calculated shifts are applied to
          the original data.  Useful functions to consider are the log of the
          image data, or 1 / data. The function is of the form func = F(data).  
          The default function ensures that the data are floats.

    Outputs
    -------
    The (y, x) position of the template in each layer in the datacube.  Output
    is two numpy arrays.

    Requires
    --------
    This function requires the "match_template" function in scikit image.

    """
    nt = datacube.shape[2]

    # Storage for the results
    keep_x = []
    keep_y = []

    # Repair the template if need be
    template = repair_nonfinite(func(template))

    # Go through each layer and perform the matching
    for t in range(0, nt):
        # Get the layer
        layer = datacube[:, :, t]

        # Repair the layer for nonfinites
        layer = repair_nonfinite(func(layer))

        # Match the template to the layer
        correlation_result = match_template_to_layer(layer, template)

        # Get the sub pixel shift of the correlation array
        y_shift_relative_to_correlation_array, \
        x_shift_relative_to_correlation_array = find_best_match_location(correlation_result)

        # Store the results
        keep_x.append(x_shift_relative_to_correlation_array)
        keep_y.append(y_shift_relative_to_correlation_array)

    return np.asarray(keep_y), np.asarray(keep_x)


#
# Coalign a mapcube
#
def coalign_mapcube(mc,
                    layer_index=0,
                    func=default_data_manipulation_function,
                    clip=False):
    """
    Co-register the layers in a mapcube according to a template taken from
    that mapcube.

    Input
    -----
    mc : a mapcube of shape (ny, nx, nt), where nt is the number of
         layers in the mapcube.

    layer_index : the layer in the mapcube from which the template will be
                  extracted.

    func: a function which is applied to the data values before the
          coalignment method is applied.  This can be useful in coalignment,
          because it is sometimes better to co-align on a function of the data
          rather than the data itself.  The calculated shifts are applied to
          the original data.  Useful functions to consider are the log of the
          image data, or 1 / data. The function is of the form func = F(data).  
          The default function ensures that the data are floats.

    clip : clip off x, y edges in the datacube that are potentially affected
            by edges effects.

    Output
    ------
    datacube : the input datacube each layer having been co-registered against
               the template.

    y_shift_keep : a one dimensional array of length nt with the pixel
                     y-displacements relative position of the template at the
                     value layer_index.  Note that y_displacement[layer_index]
                     is zero by definition.

    x_shift_keep: a one dimensional array of length nt with the pixel
                     x-displacements relative position of the template at the
                     value layer_index.  Note that x_displacement[layer_index]
                     is zero by definition.
    """
    # Size of the data
    ny = mc._maps[layer_index].shape[0]
    nx = mc._maps[layer_index].shape[1]
    nt = len(mc._maps)

    # Storage for the shifted data and the pixel shifts
    shifted_datacube = np.zeros((ny, nx, nt))
    xshift_keep = np.zeros((nt))
    yshift_keep = np.zeros((nt))

    # Calculate a template
    template = repair_nonfinite(func(mc._maps[layer_index].data[ny / 4: 3 * ny / 4,
                                         nx / 4: 3 * nx / 4]))

    for i, m in enumerate(mc._maps):
        # Get the next 2-d data array
        this_layer = func(m.data)

        # Repair any NANs, Infs, etc in the layer
        this_layer = repair_nonfinite(this_layer)

        # Calculate the correlation array matching the template to this layer
        corr = match_template_to_layer(this_layer, template)

        # Calculate the y and x shifts in pixels
        yshift, xshift = find_best_match_location(corr)

        # Keep shifts in pixels
        yshift_keep[i] = -yshift
        xshift_keep[i] = -xshift

    # Calculate shifts relative to the template layer
    yshift_keep = yshift_keep - yshift_keep[layer_index]
    xshift_keep = xshift_keep - xshift_keep[layer_index]
    
    # Shift the data
    for i, m in mc._maps:
        shifted_datacube[:, :, i] = shift(m.data, [-yshift, -xshift])

    # Clip the data if requested
    if clip:
        shifted_datacube = clip_edges(shifted_datacube, yshift_keep, xshift_keep)

    # Create a new mapcube.  Adjust the positioning information accordingly.
    new_cube = []
    for i, m in mc._maps:
        new_meta = m.meta.copy()
        new_meta['xcen'] = new_meta['xcen'] + xshift_keep[i] * m.scale['x']
        new_meta['ycen'] = new_meta['ycen'] + yshift_keep[i] * m.scale['y']
        new_map = Map(shifted_datacube[:, :, i], new_meta)

        # Store the new map in a list
        new_cube.append(new_map)

    # Return the cube and the pixel displacements
    return Map(new_cube, cube=True), yshift_keep, xshift_keep


def default_data_manipulation_function(data):
    """
    This function ensures that the data are floats.  It is the default data
    manipulation function for the coalignment method.
    """
    return 1.0 * data


#
# Remove the edges of a datacube
#
def clip_edges(datacube, y, x):
    """
    Clips off the y and x edges of a datacube according to a list of pixel
    values.  Positive pixel values will clip off the datacube at the upper end
    of the range.  Negative values will clip off values at the lower end of
    the  range.  This function is useful for removing data at the edge of
    datacubes that may be affected by shifts from solar de-rotation and
    layer co-registration, leaving a datacube unaffected by edge effects.

    Input
    -----
    datacube : a numpy array of shape (ny, nx, nt), where nt is the number of
               layers in the datacube.

    y : a numpy array of pixel values that correspond to how much to pixel
        clip values in the x-direction.

    x : a numpy array of pixel values that correspond to how much to pixel
        clip values in the y-direction.

    Output
    ------
    A datacube with edges clipd off according to the positive and negative
    ceiling values in the y and x arrays.
    """

    # Datacube shape
    ny = datacube.shape[0]
    nx = datacube.shape[1]
    nt = datacube.shape[2]

    # maximum to clip off in the x-direction
    xupper = _upper_clip(x)
    xlower = _lower_clip(x)

    # maximum to clip off in the y direction
    yupper = _upper_clip(y)
    ylower = _lower_clip(y)

    return datacube[ylower: ny - yupper - 1, xlower: nx - xupper - 1, 0: nt]


#
# Helper functions for clipping edges
#
def _upper_clip(z):
    zupper = 0
    zcond = z >= 0
    if np.any(zcond):
        zupper = np.max(np.ceil(z[zcond]))
    return zupper


def _lower_clip(z):
    zlower = 0
    zcond = z <= 0
    if np.any(zcond):
        zlower = np.max(np.ceil(-z[zcond]))
    return zlower


def match_template_to_layer(layer, template):
    """
    Calculate the correlation array that describes how well the template
    matches the layer.
    All inputs are assumed to be numpy arrays.

    Inputs
    ------
    template : a numpy array of size (N, M) where N < ny and M < nx .

    layer : a numpy array of size (ny, nx), where the first two
               dimensions are spatial dimensions.

    Outputs
    -------
    A cross-correlation array.  The values in the array range between 0 and 1.

    Requires
    --------
    This function requires the "match_template" function in scikit image.

    """
    return match_template(layer, template)


def find_best_match_location(corr):
    """
    Calculate an estimate of the location of the peak of the correlation
    result.

    Inputs
    ------
    corr : a 2-d correlation array.

    Output
    ------
    y, x : the shift amounts.  Subpixel values are possible.

    """
    # Get the index of the maximum in the correlation function
    ij = np.unravel_index(np.argmax(corr), corr.shape)
    cor_max_x, cor_max_y = ij[::-1]

    # Get the correlation function around the maximum
    array_around_maximum = corr[np.max([0, cor_max_y - 1]): np.min([cor_max_y + 2, corr.shape[0] - 1]), 
                                  np.max([0, cor_max_x - 1]): np.min([cor_max_x + 2, corr.shape[1] - 1])]
    y_shift_relative_to_maximum, x_shift_relative_to_maximum = \
    get_correlation_shifts(array_around_maximum)

    # Get shift relative to correlation array
    y_shift_relative_to_correlation_array = y_shift_relative_to_maximum + cor_max_y
    x_shift_relative_to_correlation_array = x_shift_relative_to_maximum + cor_max_x

    return y_shift_relative_to_correlation_array, x_shift_relative_to_correlation_array


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
    represents an equally spaced sampling at the locations f(-1), f(0) and
    f(1).

    Input
    -----
    An one dimensional numpy array of shape 3

    Output
    ------
    A digit, the location of the parabola maximum.

    """
    numerator = -0.5 * y.dot([-1, 0, 1])
    denominator = y.dot([1, -2, 1])
    return numerator / denominator


def repair_nonfinite(z):
    """
    Replace all the nonfinite entries in a layer with the local mean.  There is
    probably a much smarter way of doing this.
    """
    nx = z.shape[1]
    ny = z.shape[0]
    bad_index = np.where(np.logical_not(np.isfinite(z)))
    while bad_index[0].size != 0:
        by = bad_index[0][0]
        bx = bad_index[1][0]

        # x locations taking in to account the boundary
        x = bx
        if bx == 0:
            x = 1
        if bx == nx - 1:
            x = nx - 2

        # y locations taking in to account the boundary
        y = by
        if by == 0:
            y = 1
        if by == ny - 1:
            y = ny - 2

        # Get the sub array around the bad index, and find the local mean
        # ignoring nans
        subarray = z[y - 1: y + 2, x - 1: x + 2]
        z[by, bx] = np.nanmean(subarray * np.isfinite(subarray))
        bad_index = np.where(np.logical_not(np.isfinite(z)))


#
# Test functions for the coalignment code
#
def test_parabolic_turning_point():
    pass


def test_repair_nonfinite():
    pass


def test_get_correlation_shifts():
    pass