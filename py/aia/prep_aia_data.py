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
aiadata = '~/Data/AIA/shutdownfun3/disk/1.0'

# Wavelength
wave = '211'

# Output data location
output = os.path.join(os.path.expanduser('~/ts/pickle/shutdownfun3/1.0'), wave)
cubetools.makedirs(output)

# Load in the derotated data into a datacube
dc, location, savename, original_mapcube = aia_specific.rn4(os.path.join(aiadata, wave), derotate=True)

# Save the SunPy mapcube
pickle.dump(original_mapcube,open(os.path.join(output, 'full.'+wave+'.mapcube.pickle'), 'wb'))

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in original_mapcube:
    date_obs.append(parse_time(original_mapcube.header['date_obs']))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times ={"date_obs": date_obs, "time_in_seconds":time_in_seconds}


# Define regions in the datacube
regions = {'moss':[[175, 210], [115, 180]],
           'sunspot':[[135, 210], [320, 420]],
           'qs':[[135, 210], [320, 420]],
           'loopfootpoints':[[165, 245], [0, 50]],
           'full':[[0, -1], [0, -1]]}

# Save all regions
cubetools.save_region(dc, output, regions, wave, times)
