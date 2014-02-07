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
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    print 'Maximum cross correlation ', x, y

    # Fit a 2-dimensional Gaussian to the correlation peak and get the
    # displacement
    gaussian_parameters = fitgaussian(result * (result > 0.0))
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
    dc[..., t] = shifted