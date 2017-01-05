"""
Step 0

Load in the FITS files and write out a mapcube that has had the derotation
and co-alignment applied as necessary.

For each channel, the solar derotation is calculated according to the time
stamps in the FITS file headers.

Image cross-correlation is applied using the shifts calculated by applying
sunpy's image co-alignment routine to the channel indicated by the variable
base_cross_correlation_channel.  This ensures that all channels get moved the
same way, and the shifts per channel do not depend on the structure or motions
in each channel.  This assumes that the original images in each AIA channel


"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sunpy.time import parse_time
from sunpy.map import Map
from sunpy.image.coalignment import mapcube_coalign_by_match_template, calculate_match_template_shift, _default_fmap_function
from sunpy.physics.solar_rotation import mapcube_solar_derotate, calculate_solar_rotate_shift
import step0_plots
import details_study as ds

# Use base cross-correlation channel?
use_base_cross_correlation_channel = ds.use_base_cross_correlation_channel

# Create the AIA source data location
aia_data_location = ds.aia_data_location["aiadata"]

# Extend the name if cross correlation is requested
extension = ds.aia_data_location

# Locations of the output datatypes
save_locations = ds.save_locations

# Identity of the data
ident = ds.ident

# Which files in particular to put in to the mapcube
file_list_index = ds.file_list_index

# Load in the derotated data into a datacube
print('Acquiring data from ' + aia_data_location)

# Get the list of data and sort it
directory_listing = sorted(os.path.join(aia_data_location, f) for f in os.listdir(aia_data_location))
list_of_data = []
if ds.fits_level == '1.0':
    ending = '.fts'
    ending_number = -4
else:
    ending = '.fits'
    ending_number = -5

for f in directory_listing:
    if f[ending_number:] == ending:
        list_of_data.append(f)
    else:
        print('File that does not end in ".fits" detected, and not included in list = %s ' %f)

print("Number of files found = %i" % len(list_of_data))
print("Indices of files used = %s" % ds.index_string)
#
# Start manipulating the data
#
print("Loading data")
mc = Map(list_of_data[file_list_index[0]:file_list_index[1]], cube=True)
zzz
# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in mc:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}


# Solar de-rotation and cross-correlation operations will be performed relative
# to the map at this index.
layer_index = len(mc) // 2
t_since_layer_index = times["time_in_seconds"] - times["time_in_seconds"][layer_index]
filepath = os.path.join(save_locations['image'], ident + '.cross_correlation.png')
#
# Apply solar derotation
#
if ds.derotate:
    print("\nPerforming de-rotation")

    # Calculate the solar rotation of the mapcube
    print("Calculating the solar rotation shifts")
    sr_shifts = calculate_solar_rotate_shift(mc, layer_index=layer_index)

    # Plot out the solar rotation shifts
    filepath = os.path.join(save_locations['image'], ident + '.solar_derotation.%s.png' % ds.index_string)
    step0_plots.plot_shifts(sr_shifts, 'shifts due to solar de-rotation',
                            layer_index, filepath=filepath)
    filepath = os.path.join(save_locations['image'], ident + '.time.solar_derotation.%s.png' % ds.index_string)
    step0_plots.plot_shifts(sr_shifts, 'shifts due to solar de-rotation',
                            layer_index, filepath=filepath,
                            x=t_since_layer_index, xlabel='time relative to reference layer (s)')

    # Apply the solar rotation shifts
    print("Applying solar rotation shifts")
    data = mapcube_solar_derotate(mc,
                                  layer_index=layer_index, shift=sr_shifts, clip=True,
                                  order=1)
else:
    data = Map(list_of_data, cube=True)

#
# Coalign images by cross correlation
#
if ds.cross_correlate:
    if use_base_cross_correlation_channel:
        ccbranches = [ds.corename, ds.sunlocation, ds.fits_level, ds.base_cross_correlation_channel]
        ccsave_locations = ds.datalocationtools.save_location_calculator(ds.roots, ccbranches)
        ccident = ds.datalocationtools.ident_creator(ccbranches)
        ccfilepath = os.path.join(ccsave_locations['pickle'], ccident + '.cross_correlation.pkl')

        if ds.wave == ds.base_cross_correlation_channel:
            print("\nPerforming cross_correlation and image shifting")
            cc_shifts = calculate_match_template_shift(mc, layer_index=layer_index)
            print("Saving cross correlation shifts to %s" % filepath)
            f = open(ccfilepath, "wb")
            pickle.dump(cc_shifts, f)
            f.close()

            # Now apply the shifts
            data = mapcube_coalign_by_match_template(data, layer_index=layer_index, shift=cc_shifts)
        else:
            print("\nUsing base cross-correlation channel information.")
            print("Loading in shifts to due cross-correlation from %s" % ccfilepath)
            f = open(ccfilepath, "rb")
            cc_shifts = pickle.load(f)
            f.close()
            print("Shifting images")
            data = mapcube_coalign_by_match_template(data, layer_index=layer_index, shift=cc_shifts)
    else:
        print("\nCalculating cross_correlations.")
        #
        # The 131 data has some very significant shifts that may be
        # related to large changes in the intensity in small portions of the
        # data, i.e. flares.  This may be throwing the fits off.  Perhaps
        # better to apply something like a log?
        #
        if ds.wave == '131' or (ds.wave == '171' and ds.study_type == 'paper3_BLSGSM'):
            cc_func = np.sqrt
        else:
            cc_func = _default_fmap_function
        print('Data will have %s applied to it.' % cc_func.__name__)
        cc_shifts = calculate_match_template_shift(data, layer_index=layer_index, func=cc_func)
        print("Applying cross-correlation shifts to the data.")
        data = mapcube_coalign_by_match_template(data, layer_index=layer_index, shift=cc_shifts)

    # Data particulars
    dts = [(m.date - data[layer_index].date).total_seconds() for m in data]
    dts_obs = [m.date for m in data]
    x_scale = data[layer_index].scale.x
    y_scale = data[layer_index].scale.y

    # Save the cross correlation shifts
    directory = save_locations['pickle']
    filename = ident + '.cc_chifts.{:s}{:s}.{:s}.pkl'.format(
        ds.step0_output_information, ds.rn_processing, ds.index_string)
    pfilepath = os.path.join(directory, filename)
    print('Saving cross-correlation data to ' + pfilepath)
    outputfile = open(pfilepath, 'wb')
    pickle.dump(cc_shifts, outputfile)
    pickle.dump(dts, outputfile)
    pickle.dump(dts_obs, outputfile)
    pickle.dump(layer_index, outputfile)
    pickle.dump(x_scale, outputfile)
    pickle.dump(y_scale, outputfile)
    outputfile.close()

    # Plot out the cross correlation shifts
    ccx = cc_shifts['x']/x_scale
    ccy = cc_shifts['y']/y_scale
    displacement_unit = ccx.unit
    displacement = np.sqrt(ccx.value**2 + ccy.value**2)
    npwr = len(displacement)
    window = np.hanning(npwr)
    pwr_r = np.abs(np.fft.fft(displacement*window, norm='ortho'))**2
    pwr_ccx = np.abs(np.fft.fft(ccx*window, norm='ortho'))**2
    pwr_ccy = np.abs(np.fft.fft(ccy*window, norm='ortho'))**2
    freq = np.fft.fftfreq(npwr, 12.0)
    plt.close('all')
    plt.figure(1)
    pwr_start = 0
    pwr_end = npwr//2
    plt.semilogy(freq[pwr_start:pwr_end], pwr_r[pwr_start:pwr_end], label='net displacement')
    plt.semilogy(freq[pwr_start:pwr_end], pwr_ccx[pwr_start:pwr_end], label='x displacement')
    plt.semilogy(freq[pwr_start:pwr_end], pwr_ccy[pwr_start:pwr_end], label='y displacement')
    plt.axis('tight')
    plt.grid('on')
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'power ({:s})'.format((displacement_unit ** 2)._repr_latex_()))
    title = 'FFT power of cross-correlation displacement\n'
    analysis_title = 'AIA {:s}, {:n} images, FITS level={:s}\n'.format(ds.wave, npwr, ds.fits_level)
    analysis_title += 'derotation and cross-correlation image index={:n}'.format(layer_index)
    plt.title(title + analysis_title)
    plt.legend(framealpha=0.5)
    plt.tight_layout()
    filepath = os.path.join(save_locations['image'], ident + '.fft_crosscorrelation.%s.png' % ds.index_string)
    plt.savefig(filepath)

    plt.figure(2)
    plt.plot(np.arange(len(dts)), dts)
    plt.axis('tight')
    plt.axvline(layer_index, color='r', label='layer index')
    plt.axhline(0.0, color='k', linestyle=":", label='reference layer')
    plt.xlabel('sample number')
    plt.ylabel('time (s) rel. to derotation and cross-correlation image\n at {:s}'.format(str(data[layer_index].date)))
    title = 'FITS recorded observation time from initial observation\n'
    plt.title(title + analysis_title)
    plt.legend(framealpha=0.5)
    plt.tight_layout()
    filepath = os.path.join(save_locations['image'], ident + '.sampletimes.%s.png' % ds.index_string)
    plt.savefig(filepath)

    filepath = os.path.join(save_locations['image'], ident + '.cross_correlation.%s.png' % ds.index_string)
    step0_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation \n using %s' % cc_func.__name__,
                            layer_index, filepath=filepath)

    filepath = os.path.join(save_locations['image'], ident + '.time.cross_correlation.%s.png' % ds.index_string)
    step0_plots.plot_shifts(cc_shifts, 'shifts due to cross correlation \n using %s'  % cc_func.__name__,
                            layer_index, filepath=filepath,
                            x=t_since_layer_index, xlabel='time relative to reference layer (s)')
#
# Save the full dataset
#
directory = save_locations['pickle']
filename = ident + '.full_mapcube{:s}{:s}.{:s}.pkl'.format(ds.step0_output_information, ds.rn_processing, ds.index_string)
pfilepath = os.path.join(directory, filename)
print('Saving data to ' + pfilepath)
outputfile = open(pfilepath, 'wb')
pickle.dump(data, outputfile)
pickle.dump(layer_index, outputfile)
outputfile.close()
