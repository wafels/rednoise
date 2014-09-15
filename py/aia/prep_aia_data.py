"""
Load in the FITS files and write out a numpy arrays
"""
# Movie creation
#import matplotlib
#matplotlib.use("Agg")
import os
#from matplotlib import rc_file
#matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1_image_plots.rc'
#rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

import cPickle as pickle
import aia_specific

from sunpy.time import parse_time
from sunpy.cm import cm
import numpy as np
import coalign_datacube
import cubetools
from paper1 import sunday_name, label_sunday_name


# input data
dataroot = '~/Data/AIA/'
#corename = '20120923_0000__20120923_0100'
corename = 'shutdownfun3_6hr'
corename = 'shutdownfun3_6hr_3files_only'
#corename = 'shutdownfun6_6hr'
sunlocation = 'disk'
fits_level = '1.5'
wave = '193'
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

# Load in the derotated data into a datacube
print('Loading' + aia_data_location["aiadata"])
dc, ysrdisp, xsrdisp, original_mapcube = cubetools.get_datacube(aia_data_location["aiadata"],
                                              derotate=True,
                                              clip=True)
# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in original_mapcube:
    date_obs.append(parse_time(m.header['date_obs']))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": time_in_seconds}

# Cross-correlate the datacube and shave the edges
if cross_correlate:
    print('Performing cross-correlation')
    ny = dc.shape[0]
    nx = dc.shape[1]
    nt = dc.shape[2]
    layer_index = nt / 2
    ind = layer_index
    template = [[ny / 4, nx / 4], [3 * ny / 4, 3 * nx / 4]]

    # Show an image of where the cross-correlation template is
    plt.imshow(np.log(dc[:, :, layer_index]), origin='bottom', cmap=cm.get_cmap(name='sdoaia' + wave))
    plt.title(ident + ': nx=%i, ny=%i' % (nx, ny))
    plt.ylabel('y pixel (%i images)' % (nt))
    plt.xlabel('x pixel (%s)' % (str(times["date_obs"][layer_index])))
    aia_specific.plot_square([template[0][1], template[1][1]],
                             [template[0][0], template[1][0]],
                             color='w', linewidth=3)
    aia_specific.plot_square([template[0][1], template[1][1]],
                             [template[0][0], template[1][0]],
                             color='k', linewidth=1)
    plt.text(template[1][1], template[1][0], 'template', color='k', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlim(0, nx)
    plt.ylim(0, ny)
    plt.savefig(os.path.join(save_locations["image"], ident + '_cross_cor_template.png'))

    # Do the cross correlation.  Using shave=False ensures that this output
    # datacube is the same size as the input datacube and so the locations of
    # the regions will be the same relative to the size of the datacube.
    dc, xccdisp, yccdisp = coalign_datacube.coalign_datacube(dc,
                                                         template_index=template,
                                                         layer_index=layer_index,
                                                         clip=True)
else:
    layer_index = None
    ind = 0

#
# Plot the displacements in pixels
#
plt.figure(2)
plt.xlabel('time index')
plt.ylabel('displacement (pixel)')
plt.title(ident)
plt.plot(xsrdisp, label='solar rot., x')
plt.plot(ysrdisp, label='solar rot., y')
plt.plot(np.sqrt(xsrdisp ** 2 + ysrdisp ** 2), label='solar rot., D')
plt.axhline(0, color='k', linestyle='--', label='no displacement')
plt.legend(framealpha=0.3)
plt.savefig(os.path.join(save_locations["image"], ident + '_solarrotation.png'))

if cross_correlate:
    plt.figure(3)
    plt.xlabel('time index')
    plt.ylabel('displacement (pixel)')
    plt.title(ident)
    plt.plot(xccdisp, label='cross corr., x')
    plt.plot(yccdisp, label='cross corr., y')
    plt.plot(np.sqrt(xccdisp ** 2 + yccdisp ** 2), label='cross corr., D')
    plt.axvline(layer_index, label='cross corr. layer [%i]' % (layer_index), color='k', linestyle="-")
    plt.axhline(0, color='k', linestyle='--', label='no displacement')
    plt.legend(framealpha=0.3)
    plt.savefig(os.path.join(save_locations["image"], ident + '_crosscorrelation.png'))


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

if (corename == 'shutdownfun3_6hr' or corename == 'shutdownfun3_6hr_3files_only') and layer_index == nt / 2:
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
nt = dc.shape[2]
ny = dc.shape[0]
nx = dc.shape[1]

omap = original_mapcube[0]
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
plt.imshow(np.log(dc[:, :, ind]),
           origin='bottom',
           cmap=cm.get_cmap(name='sdoaia' + wave),
           extent=extent)
#plt.title(ident + ': nx=%i, ny=%i' % (nx, ny))
plt.title('AIA ' + wave + ' (%s)' % (times["date_obs"][ind].strftime('%Y/%m/%d %H:%M:%S')))
plt.ylabel('y (arcseconds)')
plt.xlabel('x (arcseconds)')
xoffset = 2
yoffset = 5
for region in regions:
    pixel_index = regions[region]
    y = pixel_index[0]
    x = pixel_index[1]
    loc1 = px2arcsec(Q, [x[0], y[0]])
    loc2 = px2arcsec(Q, [x[1], y[1]])
    aia_specific.plot_square([loc1[0], loc2[0]], [loc1[1], loc2[1]], color='w', linewidth=3)
    aia_specific.plot_square([loc1[0], loc2[0]], [loc1[1], loc2[1]], color='k', linewidth=1)
    if region == 'moss':
        plt.text(loc2[0] + xoffset, loc2[1] + yoffset, label_sunday_name[region], color='k', bbox=dict(facecolor='white', alpha=0.5), size=20)
    else:
        plt.text(loc2[0] + xoffset, loc1[1] - yoffset, label_sunday_name[region], color='k', bbox=dict(facecolor='white', alpha=0.5), size=20)
plt.xlim(lower_left[0], upper_right[0])
plt.ylim(lower_left[1], upper_right[1])
plt.savefig(os.path.join(save_locations["image"], ident + '.eps'))
"""

#
# Save all regions
#
#regions = dict(regions.items() + regions_central.items())
keys = regions.keys()
for region in keys:
    # Get the location of the region we are interested in.
    pixel_index = regions[region]
    y = pixel_index[0]
    x = pixel_index[1]
    # Region identifier name
    region_id = ident + '_' + region
    # branch location
    b = [corename, sunlocation, fits_level, wave, region]
    # Output location
    output = aia_specific.save_location_calculator(roots, b)["pickle"]
    # Output filename
    ofilename = os.path.join(output, region_id + '.datacube.pickle')
    # Open the file and write it out
    outputfile = open(ofilename, 'wb')
    pickle.dump(dc[y[0]: y[1], x[0]:x[1], :], outputfile)
    pickle.dump(times, outputfile)
    pickle.dump(pixel_index, outputfile)
    outputfile.close()
    print('Saved to ' + ofilename)
"""
"""
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    print('Starting to write movie ' + movie_output)
    # Construct the movie
    movie_output = os.path.join(movie, region_id + '.derotated_output_movie.mp4')
    fig = plt.figure(region_id)
    img = []
    for i in range(0, nt):
        img.append([plt.imshow(np.log(dc[y[0]: y[1], x[0]:x[1], i]), origin='bottom')])

    ani = animation.ArtistAnimation(fig, img, interval=20, blit=True, repeat_delay=0)
    ani.save(movie_output, writer=writer)
    print('Movie written to ' + movie_output)
"""

# Save the SunPy mapcube
#pickle.dump(original_mapcube, open(os.path.join(output, 'full.' + wave + '.mapcube.pickle'), 'wb'))


