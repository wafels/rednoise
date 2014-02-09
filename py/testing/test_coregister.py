#
# Quick program to test co-registration stuff
#
import numpy as np
import pickle
from cubetools import fitgaussian_for_coregistration
from matplotlib import pyplot as plt
# For image co-registration
# match a sample area in a larger area
from skimage.feature import match_template

# shift an array
from scipy.ndimage.interpolation import shift

# Interactive plotting
plt.ion()

# Plot an image of the regions
def plot_square(x, y, **kwargs):
    plt.plot([x[0], x[0]], [y[0], y[1]], **kwargs)
    plt.plot([x[0], x[1]], [y[1], y[1]], **kwargs)
    plt.plot([x[1], x[1]], [y[0], y[1]], **kwargs)
    plt.plot([x[0], x[1]], [y[0], y[0]], **kwargs)

region = 'moss'

# data
#filename = '/home/ireland/ts/pickle/shutdownfun3_1hr/disk/1.0/171/moss.171.datacube.pickle'
filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/' + \
region + '/20120923_0000__20120923_0100_disk_1.5_171_' + region + '.datacube.pickle'
#filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/qs/20120923_0000__20120923_0100_disk_1.5_171_qs.datacube.pickle'
#filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/sunspot/20120923_0000__20120923_0100_disk_1.5_171_sunspot.datacube.pickle'
#filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/loopfootpoints/20120923_0000__20120923_0100_disk_1.5_171_loopfootpoints.datacube.pickle'

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
len_y = ny / 2.5
len_x = nx / 2.5
template_y = [center[0] - len_y, center[0] + len_y]
template_x = [center[1] - len_x, center[1] + len_x]


# use the log of the data
# get rid of the extreme values?

# Median template
template = np.log(np.median(dc[template_y[0]:template_y[1],
                        template_x[0]:template_x[1], :], axis= 2))
# Mean template
#template = np.mean(dc[template_y[0]:template_y[1],
#                      template_x[0]:template_x[1], :], axis= 2)

# Geometric mean template
template = np.mean(np.log10(dc[template_y[0]:template_y[1],
                                     template_x[0]:template_x[1], :]), axis= 2)



keep_x = []
keep_y = []

# Go through each layer and perform the matching
for t in range(1, nt):
    print ' '
    print 'Layer ' + str(t)
    # The previous layer is the reference layer
    layer = dc[:, :, t]
    
    # get a template for the current layer
    #current_layer = dc[:, :, t]
    #template = current_layer[template_y[0]:template_y[1],
    #                         template_x[0]:template_x[1]]

    # Match the current template to the previous layer
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
    gaussian_parameters = fitgaussian_for_coregistration(result * (result > 0.0))

    # Calculate the offset - could be less than one pixel.
    ydiff = 0
    xdiff = 0
    print 'Gaussian parameters : ' + str(gaussian_parameters)
    keep_x.append(gaussian_parameters[1])
    keep_y.append(gaussian_parameters[2])

    # Shift the current layer
    #if np.abs(ydiff) >= diff_limit and np.abs(xdiff) >= diff_limit:
    #    shifted = shift(current_layer, [-ydiff, -xdiff])
    #else:
    #    shifted = current_layer

    # Store it back in the datacube.
    #dc[..., t] = shifted

plt.figure(1)
plt.imshow(dc[:, :, register_index])
plot_square(template_x, template_y, color='w', linewidth=3)
plot_square(template_x, template_y, color='k', linewidth=1)
plt.title(region)



plt.figure(2)
kx = np.asarray(keep_x - keep_x[0])
ky = np.asarray(keep_y - keep_y[0])
plt.plot(kx, label='x displacement')
plt.plot(ky, label='y displacement')
plt.plot(np.sqrt(kx ** 2 + ky ** 2), label ='total displacement')
plt.xlabel('time index')
plt.ylabel('pixel displacement (relative to initial)')
plt.title(region)
plt.legend()
