import numpy as np

# Shift an image by a given amount - subpixel shifts are permitted
from scipy.ndimage.interpolation import shift

#
from coalign_mapcube import default_data_manipulation_function, clip_edges, calculate_shift


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
    # Size of the data
    ny = datacube.shape[0]
    nx = datacube.shape[1]
    nt = datacube.shape[2]

    # Storage for the shifted data and the pixel shifts
    xshift_keep = np.zeros((nt))
    yshift_keep = np.zeros((nt))

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

    # Apply the data manipulation function
    template = func(template)

    for i in range(0, nt):
        # Get the next 2-d data array
        this_layer = func(datacube[:, :, i])

        # Calculate the y and x shifts in pixels
        yshift, xshift = calculate_shift(this_layer, template)

        # Keep shifts in pixels
        yshift_keep[i] = yshift
        xshift_keep[i] = xshift

    # Calculate shifts relative to the template layer
    yshift_keep = yshift_keep - yshift_keep[layer_index]
    xshift_keep = xshift_keep - xshift_keep[layer_index]

    # Shift the data
    shifted_datacube = shift_datacube_layers(datacube, -yshift_keep, -xshift_keep)

    if clip:
        return clip_edges(shifted_datacube, yshift_keep, xshift_keep), yshift_keep, xshift_keep
    else:
        return shifted_datacube, yshift_keep, xshift_keep


#
# Shift a datacube.  Useful for coaligning images and performing solar
# derotation.
#
def shift_datacube_layers(datacube, yshift, xshift):
    ny = datacube.shape[0]
    nx = datacube.shape[1]
    nt = datacube.shape[2]
    shifted_datacube = np.zeros((ny, nx, nt))
    for i in range(0, nt):
        shifted_datacube[:, :, i] = shift(datacube[:, :, i], [yshift[i], xshift[i]])

    return shifted_datacube
