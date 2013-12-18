"""
Load in the FITS files and write out a numpy arrays
"""
# Movie creation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cPickle as pickle
import aia_specific
import os
from sunpy.time import parse_time
import cubetools
import numpy as np


# input data
dataroot = '~/Data/AIA/'
corename = 'shutdownfun3_6hr'
sunlocation = 'disk'
fits_level = '1.5'
wave = '131'

# Pickle file storage
pickleroot = '~/ts/pickle/'

# Movie file storage
movieroot = '~/ts/movies/'

# Where is the data
aiadata = os.path.join(os.path.expanduser(dataroot), corename, sunlocation, fits_level, wave)

# Output data location
output = os.path.join(os.path.expanduser(pickleroot), corename, sunlocation, fits_level, wave)
cubetools.makedirs(output)

# Movie location
movie = os.path.join(os.path.expanduser(movieroot), corename, sunlocation, fits_level, wave)
cubetools.makedirs(movie)

# Load in the derotated data into a datacube
dc, location, savename, original_mapcube = aia_specific.rn4(aiadata, derotate=True)

# Identifier
id = corename + '_' + sunlocation + '_' + fits_level + '_' + wave

# Shape of the datacube
nt = dc.shape[2]

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in original_mapcube:
    date_obs.append(parse_time(m.header['date_obs']))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": time_in_seconds}


# Define regions in the datacube
regions = {'moss': [[175, 210], [115, 180]],
           'sunspot': [[135, 210], [320, 420]],
           'qs': [[150, 200], [520, 570]],
           'loopfootpoints': [[165, 245], [0, 50]]}

# Save all regions
# cubetools.save_region(dc, output, regions, wave, times)

keys = regions.keys()
for region in keys:
    pixel_index = regions[region]
    y = pixel_index[0]
    x = pixel_index[1]
    #filename = region + '.' + wave + '.' + str(pixel_index) + '.datacube.pickle'
    region_id = region + '.' + id
    ofilename = os.path.join(output, region_id + '.datacube.pickle')

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
pickle.dump(original_mapcube, open(os.path.join(output, 'full.' + wave + '.mapcube.pickle'), 'wb'))


