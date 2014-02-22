import numpy as np
import sunpy
import matplotlib.pyplot as plt
from scipy.io import readsav
import os
from sunpy.coords.util import rot_hpc
from sunpy.map import Map
import tsutils
import pickle 

# For image co-registration
from skimage.feature import match_template

# Shift an image by a given amount - subpixel shifts are permitted
from scipy.ndimage.interpolation import shift


def derotated_datacube_from_mapcube(maps, ref_index=0, shave=False):
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

    # Optionally shave the datacube to remove data that may be affected by edge
    # effects due to solar derotation.
    if shave:
        return shave_edges(datacube, ydiff, xdiff)
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


def get_datacube(path, derotate=False):
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
# Coalign a mapcube
#
def coalign_mapcube(mc, layer_index=0):
    """
    
    """
    # Size of the data
    nt = len(mc)
    ny = mc.maps[layer_index].shape[0]
    nx = mc.maps[layer_index].shape[1]


    # Storage for the new datacube
    new_cube = []

    # Calculate a template
    template = mc.maps[layer_index].data[ny / 4: 3 * ny / 4,
                                         nx / 4: 3 * nx / 4]

    for i in range(0, nt):
        # Get the next 2-d data array
        this_layer = mc.maps[i].data
        
        # Calculate the correlation array matching the template to this layer
        corr = match_template_to_layer(this_layer, template)
        
        # Calculate the y and x shifts
        yshift, xshift = find_best_match_location(corr)
        
        # Shift the layer
        new_data = shift(this_layer, [-yshift, -xshift])

        # Create a new map.  Adjust the positioning information accordingly.
        new_map = Map(new_data, new_meta)

        # Store the new map in a list
        new_cube.append(new_map)

    return Map(new_cube, cube=True)

#
# Remove the edges of a datacube 
#
def shave_edges(datacube, y, x):
    """
    Shaves off the y and x edges of a datacube according to a list of pixel
    values.  Positive pixel values will shave off the datacube at the upper end
    of the range.  Negative values will shave off values at the lower end of
    the  range.  This function is useful for removing data at the edge of
    datacubes that may be affected by shifts from solar de-rotation and
    layer co-registration, leaving a datacube unaffected by edge effects.

    Input
    -----
    datacube : a numpy array of shape (ny, nx, nt), where nt is the number of
               layers in the datacube.

    y : a numpy array of pixel values that correspond to how much to pixel
        shave values in the x-direction.

    x : a numpy array of pixel values that correspond to how much to pixel
        shave values in the y-direction.

    Output
    ------
    A datacube with edges shaved off according to the positive and negative
    ceiling values in the y and x arrays.
    """
    
    # Datacube shape
    ny = datacube.shape[0]
    nx = datacube.shape[1]
    nt = datacube.shape[2]

    # maximum to shave off in the x-direction  
    xupper = np.max(np.ceil(x[x >= 0]))
    xlower = np.max(np.ceil(-x[x <= 0]))

    # maximum to shave off in the y direction
    yupper = np.max(np.ceil(y[y >= 0]))
    ylower = np.max(np.ceil(-y[y <= 0]))

    return datacube[ylower: ny - yupper - 1, xlower: nx - xupper - 1, 0: nt]

#
# Shift a datacube according to calculated co-registration displacements
#
def coregister_datacube(datacube, layer_index=0, template_index=None,
                        shave=False):
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
                     to None, then the default template is used.

    shave : shave off x, y edges in the datacube that are potentially affected
            by edges effects.

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
        template = datacube[template_index[0][0]:template_index[0][1],
                            template_index[1][0]:template_index[1][1],
                            layer_index]

    # Calculate the displacements of each layer in the datacube relative to the
    # known position of a template
    y_displacement, x_displacement = calculate_coregistration_displacements(template, datacube)

    # Displacement relative to the layer_index
    y_displacement = y_displacement - y_displacement[layer_index]
    x_displacement = x_displacement - x_displacement[layer_index]

    # Shift each layer of the datacube the required amounts
    datacube = shift_datacube_layers(datacube, y_displacement, x_displacement)

    if shave:
        return shave_edges(datacube, y_displacement, x_displacement), y_displacement, x_displacement
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

    # Go through each layer and perform the matching
    for t in range(0, nt):
        # Get the layer
        layer = datacube[:, :, t]
    
        # Match the template to the layer
        correlation_result = match_template_to_layer(layer, template)
    
        # Get the sub pixel shift of the correlation array
        y_shift_relative_to_correlation_array, \
        x_shift_relative_to_correlation_array = find_best_match_location(correlation_result)

        # Store the results
        keep_x.append(x_shift_relative_to_correlation_array)
        keep_y.append(y_shift_relative_to_correlation_array)

    return np.asarray(keep_y), np.asarray(keep_x)


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
