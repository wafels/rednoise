"""
Time-series analysis

(1) Specifies a data directory
(2) Returns a de-rotated data cube from the data in that directory
(3) Generates and saves the spatial locations in that data-cube.
(4) Performs analyses on each of those time-series

"""
# Test 6: Posterior predictive checking
import numpy as np
from matplotlib import pyplot as plt
import pickle
import rnfit2
import os
#from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
import aia_specific
plt.ion()


#if False:
#    location = '~/Data/oscillations/mcateer/outgoing3/AR_A.sav'
#    maindir = os.path.expanduser(location)
#    print('Loading data from ' + maindir)
#    wave = '???'
#    idl = readsav(maindir)
#    dc = np.swapaxes(np.swapaxes(idl['region_window'], 0, 2), 0, 1)
#else:

savedir = '~/ts/pickle/'
fitsdata = '~/Data/AIA/shutdownfun3/'
#save_location = '/Users/ireland/ts/pickle/_Users_ireland_Data_AIA_shutdownfun3_171'
#
if fitsdata is not None:
    wave = '171'
    AR = True
    if AR:
        Xrange = [0, 300]
        Yrange = [150, 250]
        type = 'AR'
    else:
        Xrange = [550, 650]
        Yrange = [20, 300]
        type = 'QS'
    
    # Get the data
    dc, location, xyrange  = aia_specific.rn4(wave, fitsdata, Xrange=Xrange,
                                    Yrange=Yrange, derotate=True)
    
    # Convert the location into an identifier
    savename = location.replace('/', '_') + '_' + xyrange
    
    # Create the save subdirectory
    save_location = os.path.join(os.path.expanduser(savedir), savename)
    if not(os.path.exists(save_location)):
        os.makedirs(save_location)
    
    # Create the pixel locations
    pixel_locations = aia_specific.get_pixel_locations(iput=dc.shape)
    
    # Get a list of time-series
    tslist = aia_specific.get_tslist(dc, pixel_locations, name=savename)

    # Save the results
    pickle.dump(dc, open(os.path.join(save_location, 'datacube.pickle'), 'wb'))
    pickle.dump(pixel_locations, open(os.path.join(save_location, 'pixel_locations.pickle'), 'wb'))
    pickle.dump(tslist, open(os.path.join(save_location, 'tslist.pickle'), 'wb'))
else:
    tslist = pickle.load(open(os.path.join(save_location, 'tslist.pickle'), 'rb'))

# Go through all the time series

# Storage for the least-square fit results
lstsqr = []

# storage for the AR(1) results
ar1 = []

for ts in tslist:

    # The positive part of the power spectrum
    ps = [ts.pfreq / ts.pfreq[0], ts.ppower]

    # An estimate for the values of the power spectrum parameters
    estimate = [ts.ppower[0], 2, np.mean(ts.ppower[-10:-1])]

    # Do the least-squares fit
    answer = rnfit2.Do_LSTSQR((ps,)).okgo(p0=(estimate,))
    lstsqr.append(answer.results[0])

    # Do the AR(1) fit
    #ar1.append(rnfit2.Do_AR1((ts.data,)))

# Save the results
pickle.dump(lstsqr, open(os.path.join(save_location, 'lstsqr.pickle'), 'wb'))
