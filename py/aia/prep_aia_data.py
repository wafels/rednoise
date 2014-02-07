"""
Load in the FITS files and write out a numpy arrays
"""
# Movie creation
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import cPickle as pickle
import aia_specific
import os
from sunpy.time import parse_time
from sunpy.cm import cm
import numpy as np


# input data
dataroot = '~/Data/AIA/'
#corename = '20120923_0000__20120923_0100'
corename = 'shutdownfun6_6hr'
sunlocation = 'disk5'
fits_level = '1.0'
wave = '131'
correlate = False

# Create the branches in order
branches = [corename, sunlocation, fits_level, wave]

# Create the AIA source data location
aia_data_location = aia_specific.save_location_calculator({"aiadata": dataroot},
                                             branches)

# Create the locations of where we will store output
if correlate:
    extension = '_correlated'
else:
    extension = ''

roots = {"pickle": '~/ts/pickle' + extension,
         "image": '~/ts/img' + extension,
         "movie": '~/ts/movies' + extension}
save_locations = aia_specific.save_location_calculator(roots, branches)

ident = aia_specific.ident_creator(branches)

# Load in the derotated data into a datacube
dc, original_mapcube = aia_specific.rn4(aia_data_location["aiadata"],
                                        derotate=True, correlate=correlate)

# Shape of the datacube
nt = dc.shape[2]
ny = dc.shape[0]
nx = dc.shape[1]

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in original_mapcube:
    date_obs.append(parse_time(m.header['date_obs']))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": time_in_seconds}


# Define regions in the datacube
# y locations are first, x locations second

if corename == 'shutdownfun6_6hr':
    regions = {'highlimb': [[100, 150], [50, 150]],
               'lowlimb': [[100, 150], [200, 250]],
               'crosslimb': [[100, 150], [285, 340]],
               'loopfootpoints1': [[90, 155], [515, 620]],
               'loopfootpoints2': [[20, 90], [793, 828]],
               'moss': [[45, 95], [888, 950]]}

if corename == 'shutdownfun3_6hr' or corename == '20120923_0000__20120923_0100':
    regions = {'moss': [[175, 210], [115, 180]],
               'sunspot': [[125, 200], [270, 370]],
               'qs': [[150, 200], [520, 570]],
               'loopfootpoints': [[165, 245], [0, 50]]}
#
# Plot an image of the data with the subregions overlaid and labeled
#
plt.figure(1)
ind = 0
plt.imshow(np.log(dc[:, :, ind]), origin='bottom', cmap=cm.get_cmap(name='sdoaia' + wave))
plt.title(ident)
plt.ylabel('y pixel (%i images)' % (nt))
plt.xlabel('x pixel (%s)' % (str(times["date_obs"][ind])))
for region in regions:
    pixel_index = regions[region]
    y = pixel_index[0]
    x = pixel_index[1]
    aia_specific.plot_square(x, y, color='w', linewidth=3)
    aia_specific.plot_square(x, y, color='k', linewidth=1)
    plt.text(x[1], y[1], region, color='k', bbox=dict(facecolor='white', alpha=0.5))
plt.xlim(0, nx)
plt.ylim(0, ny)
plt.savefig(os.path.join(save_locations["image"], ident + '.png'))
plt.close('all')

#
# Save all regions
# Used to be cubetools.save_region(dc, output, regions, wave, times)
# but that took up too much memory
#
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


