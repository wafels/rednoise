import numpy as np

# Shift an image by a given amount - subpixel shifts are permitted
from scipy.ndimage.interpolation import shift

# 
from coalign_mapcube import default_data_manipulation_function, clip_edges, repair_nonfinite, match_template_to_layer, find_best_match_location

#
# Coalign a datacube.  Can be useful to have this functionality when you just
# want to deal with datacubes and not the more complex maps.
#
# Shift a datacube according to calculated co-registration displacements
#
def coalign_datacube(datacube, layer_index=0, template_index=None,
                        clip=False, func=default_data_manipulation_function):
    """
    Co-align the layers in a datacube by finding where a template best matches
    each layer in the datacube.

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
