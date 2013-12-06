"""
Load in the FITS files and write out a numpy arrays
"""
# Test 6: Posterior predictive checking
import cPickle as pickle
import aia_specific
import os

"""

"""
# inout data
aiadata = '~/Data/AIA/shutdownfun3/disk/1.0'

# wavelength
wave = '131'

# output data
output = os.path.join(os.path.expanduser('~/ts/pickle/shutdownfun3/1.0'), wave)
if not os.path.isdir(output):
    os.makedirs(output)

dc, location, savename, original_mapcube = aia_specific.rn4(os.path.join(aiadata, wave), derotate=True)

kkdasd = lkasda

# moss
pickle.dump(dc[175:210,115:180,:], open(os.path.join(output, 'moss.'+wave+'.datacube.pickle'), 'wb'))

# sunspot
pickle.dump(dc[135:210, 320:420, :], open(os.path.join(output, 'sunspot.'+wave+'.datacube.pickle'), 'wb'))

# QS
pickle.dump(dc[150:200, 650:700, :], open(os.path.join(output, 'qs.'+wave+'.datacube.pickle'), 'wb'))

# Loop footpoints
pickle.dump(dc[165:245, 0:50, :], open(os.path.join(output, 'loopfootpoints.'+wave+'.datacube.pickle'), 'wb'))

# SunPy mapcibe
pickle.dump(original_mapcube,open(os.path.join(output, 'full.'+wave+'.mapcube.pickle'), 'wb'))

# full data
pickle.dump(dc, open(os.path.join(output, 'full.'+wave+'.datacube.pickle'), 'wb'))
