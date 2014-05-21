"""
Load in the FITS files and write out a numpy arrays
"""
# Movie creation
#import matplotlib
#matplotlib.use("Agg")
import os
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1_image_plots.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

import aia_specific

from sunpy.cm import cm
import numpy as np
from paper1 import sunday_name
import sunpy.map as Map


# input data
dataroot = '~/Data/AIA/'
#corename = '20120923_0000__20120923_0100'
corename = 'shutdownfun3_6hr'
#corename = 'shutdownfun6_6hr'
sunlocation = 'disk'
fits_level = '1.5'
wave = '171'
cross_correlate = True


# Create the branches in order
branches = [corename, sunlocation, fits_level, wave]

# Create the AIA source data location
aia_data_location = aia_specific.save_location_calculator({"aiadata": dataroot},
                                             branches)

if cross_correlate:
    extension = '_cc_final'
else:
    extension = ''

roots = {"pickle": '~/ts/pickle' + extension,
         "image": '~/ts/img' + extension,
         "movie": '~/ts/movies' + extension}
save_locations = aia_specific.save_location_calculator(roots, branches)

ident = aia_specific.ident_creator(branches)

# Load in the single file we need
nt = 1800
layer_index = nt / 2
if wave == '171':
    filename = 'AIA20120923_025959_0171.fits'
    omap = Map(os.path.join(aia_data_location[''], filename))
if wave == '193':
    filename = 'AIA20120923_030006_0193.fits'
    omap = Map(os.path.join(aia_data_location[''], filename))

# Define regions in the datacube
# y locations are first, x locations second

if corename == 'shutdownfun6_6hr':
    print('Using %s and %f' % (corename, layer_index))
    regions = {'highlimb': [[100, 150], [50, 150]],
               'lowlimb': [[100, 150], [200, 250]],
               'crosslimb': [[100, 150], [285, 340]],
               'loopfootpoints1': [[90, 155], [515, 620]],
               'loopfootpoints2': [[20, 90], [793, 828]],
               'moss': [[45, 95], [888, 950]]}
#    regions_central = define_central_data(regions)

if corename == 'shutdownfun3_6hr' and layer_index == nt / 2:
    print('Using %s and %f' % (corename, layer_index))
    regions = {'moss': [[175, 210], [140, 200]],
               'sunspot': [[125, 200], [250, 350]],
               'qs': [[60, 110], [500, 550]],
               'loopfootpoints': [[110, 160], [10, 50]]}
#    regions_central = define_central_data(regions)

if corename == '20120923_0000__20120923_0100' or corename == 'shutdownfun3_1hr' :
    regions = {'moss': [[175, 210], [115, 180]],
               'sunspot': [[125, 200], [320, 420]],
               'qs': [[60, 110], [600, 650]],
               'loopfootpoints': [[165, 245], [10, 50]]}
#    regions_central = define_central_data(regions)

"""
def define_central_data(regions):
    # For each of the regions, calculate the location in the middle of the region
    new_regions = {}
    keys = regions.keys()
    for k in keys:
        yloc = regions[k][0]
        xloc = regions[k][1]
        y = np.rint( 0.5 * (yloc[0] + yloc[1]))
        x = np.rint( 0.5 * (xloc[0] + xloc[1]))
        new_key = 'k' + '_central'
        new_regions[new_key] = [[y, y], [x, x]]
    return new_regions
"""
#
# Plot an image of the data with the subregions overlaid and labeled
#

# Shape of the datacube
ny = omap.data.shape[0]
nx = omap.data.shape[1]

cXY = [omap.meta['xcen'], omap.meta['ycen']]
dXY = [omap.meta['cdelt1'], omap.meta['cdelt2']]
nXY = [omap.data.shape[1], omap.data.shape[0]]

Q = {'cen': cXY, 'd': dXY, 'n': nXY}


def px2arcsec(Q, A):
    cen = Q['cen']
    d = Q['d']
    n = Q['n']
    xcen = cen[0]
    ycen = cen[1]
    dx = d[0]
    dy = d[1]
    nx = n[0]
    ny = n[1]
    llx = xcen - 0.5 * dx * nx
    lly = ycen - 0.5 * dy * ny
    xpixel = A[0]
    ypixel = A[1]
    x_arcsec = llx + dx * xpixel
    y_arcsec = lly + dy * ypixel
    return [x_arcsec, y_arcsec]

lower_left = px2arcsec(Q, [0, 0])
upper_right = px2arcsec(Q, [nx - 1, ny - 1])
extent = [lower_left[0], upper_right[0], lower_left[1], upper_right[1]]


plt.figure(1)
plt.imshow(np.log(omap.data),
           origin='bottom',
           cmap=cm.get_cmap(name='sdoaia' + wave),
           extent=extent)
#plt.title(ident + ': nx=%i, ny=%i' % (nx, ny))
plt.title('AIA ' + wave + ' (%s)' % (omap.meta['date'].strftime('%Y/%m/%d %H:%M:%S')))
plt.ylabel('y (arcseconds)')
plt.xlabel('x (arcseconds)')
for region in regions:
    pixel_index = regions[region]
    y = pixel_index[0]
    x = pixel_index[1]
    loc1 = px2arcsec(Q, [x[0], y[0]])
    loc2 = px2arcsec(Q, [x[1], y[1]])
    aia_specific.plot_square([loc1[0], loc2[0]], [loc1[1], loc2[1]], color='w', linewidth=3)
    aia_specific.plot_square([loc1[0], loc2[0]], [loc1[1], loc2[1]], color='k', linewidth=1)
    plt.text(loc2[0], loc2[1], sunday_name[region], color='k', bbox=dict(facecolor='white', alpha=0.5))
plt.xlim(lower_left[0], upper_right[0])
plt.ylim(lower_left[1], upper_right[1])
plt.savefig(os.path.join(save_locations["image"], ident + '.eps'))

