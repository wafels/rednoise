#
# Quick program to test co-registration stuff
#
import numpy as np
import pickle
from cubetools import fitgaussian
from matplotlib import pyplot as plt
# For image co-registration
# match a sample area in a larger area
from skimage.feature import match_template

# shift an array
from scipy.ndimage.interpolation import shift

# Interactive plotting
plt.ion()

# data
filename = '/home/ireland/ts/pickle/shutdownfun3_1hr/disk/1.0/171/moss.171.datacube.pickle'
register_index = 0
diff_limit = 0.01

print 'Loading ' + filename
pkl_file = open(filename, 'rb')
dc = pickle.load(pkl_file)
pkl_file.close()

"""
Co-register a series of layers in a datacube
"""
# data cube shape
ny = dc.shape[0]
nx = dc.shape[1]
nt = dc.shape[2]

# Define a template at the registration layer
center = [ny / 2, nx / 2]
len_y = ny / 5
len_x = nx / 5
template_y = [center[0] - len_y, center[0] + len_y]
template_x = [center[1] - len_x, center[1] + len_x]
template = dc[template_y[0]:template_y[1],
              template_x[0]:template_x[1],
              register_index]

keep_x = []
keep_y = []

# Go through each layer and perform the matching
for t in range(1, nt):
    print ' '
    print 'Layer ' + str(t)
    # Get the layer
    layer = dc[:, :, t]

    # Get the cross correlation function
    result = match_template(layer, template)
    
    # Get the index of the maximum in the correlation function
    ij = np.unravel_index(np.argmax(result), result.shape)
    cor_max_x, cor_max_y = ij[::-1]
    print 'Maximum cross correlation ', cor_max_x, cor_max_y

    # Fit a 2-dimensional Gaussian to the correlation peak.  Use only
    # the positively correlated parts of the cross-correlation
    # function.  We also ensure that the initial estimate of the location
    # of the Gaussian peak is right on top of the maximum of the cross-
    # correlation function.  This ensures that the final fit will not wander
    # too far from the peak of the cross-correlation function, which is what we
    # expect for image data that has already been corrected for solar rotation.
    gaussian_parameters = fitgaussian(result * (result > 0.0),
                                      estimate=[None, cor_max_x, cor_max_y, None, None])

    # Calculate the offset - could be less than one pixel.
    ydiff = 0
    xdiff = 0
    print 'Gaussian parameters : ' + str(gaussian_parameters)
    keep_x.append(gaussian_parameters[1])
    keep_y.append(gaussian_parameters[2])

    # Shift the layer
    if np.abs(ydiff) >= diff_limit and np.abs(xdiff) >= diff_limit:
        shifted = shift(layer, [-ydiff, -xdiff])
    else:
        shifted = layer

    # Store it back in the datacube.
    dc[..., t] = shifted