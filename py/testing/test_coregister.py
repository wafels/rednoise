#
# Quick program to test co-registration stuff
#
import numpy as np
import pickle
from cubetools import coregister_datacube
from matplotlib import pyplot as plt

# Interactive plotting
plt.ion()


# Plot an image of the regions
def plot_square(x, y, **kwargs):
    plt.plot([x[0], x[0]], [y[0], y[1]], **kwargs)
    plt.plot([x[0], x[1]], [y[1], y[1]], **kwargs)
    plt.plot([x[1], x[1]], [y[0], y[1]], **kwargs)
    plt.plot([x[0], x[1]], [y[0], y[0]], **kwargs)

region = 'sunspot'
#region = 'qs'
#region = 'loopfootpoints'
#region = 'moss'

# data
#filename = '/home/ireland/ts/pickle/shutdownfun3_1hr/disk/1.0/171/moss.171.datacube.pickle'
filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/' + \
region + '/20120923_0000__20120923_0100_disk_1.5_171_' + region + '.datacube.pickle'
#filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/qs/20120923_0000__20120923_0100_disk_1.5_171_qs.datacube.pickle'
#filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/sunspot/20120923_0000__20120923_0100_disk_1.5_171_sunspot.datacube.pickle'
#filename = '/Users/ireland/ts/pickle/20120923_0000__20120923_0100/disk/1.5/171/loopfootpoints/20120923_0000__20120923_0100_disk_1.5_171_loopfootpoints.datacube.pickle'
#filename = '/home/ireland/ts/pickle/shutdownfun3_6hr/disk/1.5/171/qs/shutdownfun3_6hr_disk_1.5_171_qs.datacube.pickle'

layer_index = 150
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


dc, keep_x, keep_y = coregister_datacube(np.log(dc),
                                         layer_index=nt / 2,
                                         template_index=[template_y, template_x])


#
# The data at the layer_index
#
plt.figure(1)
plt.imshow(dc[:, :, layer_index])
plot_square(template_x, template_y, color='w', linewidth=3)
plot_square(template_x, template_y, color='k', linewidth=1)
plt.title(region)



plt.figure(2)
kx = np.asarray(keep_x)
ky = np.asarray(keep_y)
plt.plot(kx, label='x displacement')
plt.plot(ky, label='y displacement')
plt.plot(np.sqrt(kx ** 2 + ky ** 2), label ='total displacement')
plt.xlabel('time index')
plt.ylabel('pixel displacement (relative to template layer index)')
plt.axvline(layer_index, color='k', linestyle=':', linewidth=3, label='template layer index')
plt.axhline(0, color='k')
plt.title(region)
plt.legend()
