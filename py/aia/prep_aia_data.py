"""
Load in the FITS files and write out a numpy arrays
"""
# Test 6: Posterior predictive checking
import cPickle as pickle
import aia_specific
import os
from sunpy.time import parse_time
import cubetools
"""

"""
# input data
dataroot = '~/Data/AIA/'
corename = 'shutdownfun3_6hr'
location = 'disk'
level = '1.0'
wave = '131'

# Pickle file storage
pickleroot = '~/ts/pickle/'

# Where is the data
aiadata = os.path.join(os.path.expanduser(dataroot), corename, location, level, wave)

# Output data location
output = os.path.join(os.path.expanduser(pickleroot), corename, location, level, wave)
cubetools.makedirs(output)

# Load in the derotated data into a datacube
dc, location, savename, original_mapcube = aia_specific.rn4(aiadata, derotate=True)

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
    ofilename = os.path.join(output, region + '.' + wave + '.datacube.pickle')

    outputfile = open(ofilename, 'wb')
    pickle.dump(dc[y[0]: y[1], x[0]:x[1], :], outputfile)
    pickle.dump(times, outputfile)
    pickle.dump(pixel_index, outputfile)
    outputfile.close()
    print('Saved to ' + ofilename)

# Save the SunPy mapcube
pickle.dump(original_mapcube, open(os.path.join(output, 'full.' + wave + '.mapcube.pickle'), 'wb'))
