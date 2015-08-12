"""
Step 1

Load in the de-rotated and co-aligned map cube, and make some cutouts that
focus on specific regions of interest.
"""

import os

import cPickle as pickle

from sunpy.time import parse_time
import numpy as np
from matplotlib.patches import Rectangle
import astropy.units as u

import study_details as sd
import step1_plots

# Load in the derotated data into a datacube
directory = sd.save_locations['pickle']
filename = sd.ident + '.full_mapcube.pkl'
print('Acquiring mapcube from ' + sd.save_locations['pickle'])
print('Filename = ' + filename)
outputfile = open(os.path.join(directory, filename), 'rb')
mc = pickle.load(outputfile)
layer_index = pickle.load(outputfile)
outputfile.close()

# Get the date and times from the original mapcube
date_obs = []
time_in_seconds = []
for m in mc:
    date_obs.append(parse_time(m.date))
    time_in_seconds.append((date_obs[-1] - date_obs[0]).total_seconds())
times = {"date_obs": date_obs, "time_in_seconds": np.asarray(time_in_seconds)}

# Layer on which all de-rotations, etc are based on.
mc_layer = mc[layer_index]

#
# Create some time-series for further analysis
#
if sd.sunlocation == 'disk':
    #
    # Define the number of regions and their extents.  Use Helio-projective
    # co-ordinates.  These co-ordinates should be chosen in reference to the
    # time of the reference layer (layer_index).
    #
    """
    regions = {"sunspot": {"llx": -335.0*u.arcsec, "lly": 0*u.arcsec, "width": 40*u.arcsec, "height": 32*u.arcsec},
               "loop footpoints": {"llx": -492*u.arcsec, "lly": 0*u.arcsec, "width": 23*u.arcsec, "height": 22*u.arcsec},
               "quiet Sun": {"llx": -200*u.arcsec, "lly": -45*u.arcsec, "width": 15*u.arcsec, "height": 26*u.arcsec},
               "moss": {"llx": -400*u.arcsec, "lly": 25*u.arcsec, "width": 45*u.arcsec, "height": 25*u.arcsec}}
    """
    regions = {"most_of_fov": {"llx": -530.0*u.arcsec, "lly": -100*u.arcsec, "width": 340*u.arcsec, "height": 200*u.arcsec}}

    # Rectangular patches
    for region in regions:
        # Next region
        R = regions[region]

        # Define a matplotlib rectangular patch to show the region on a map
        new_rectangle = Rectangle((R['llx'].value, R['lly'].value),
                                  R['width'].value, R['height'].value,
                                  label=region, fill=False,
                                  facecolor='b', edgecolor='r', linewidth=2)

        # Store the information about the rectangle
        R["patch"] = new_rectangle
        R["path"] = new_rectangle.get_path().transformed(transform=new_rectangle.get_transform())
        R['label_offset'] = {"x": 0.0, "y": -10}

        # Information about the ranges
        R['xrange'] = [R['llx'].value, R['llx'].value + R['width'].value]
        R['yrange'] = [R['lly'].value, R['lly'].value + R['height'].value]
        llxy_pixel = mc_layer.data_to_pixel(R['llx'], R['lly'])
        R['llxy_pixel'] = np.floor([llxy_pixel[0].value, llxy_pixel[1].value])

        # Small changes in the plate scale in each AIA channel can mean that
        # the height and widths of each region depend on the data. To mitigate
        # against this we fix the size of the plate scale.
        fixed_aia_scale = sd.fixed_aia_scale#{'x': 0.6, 'y': 0.6}
        R['width_pixel'] = np.floor(R['width'] / fixed_aia_scale['x']).value
        R['height_pixel'] = np.floor(R['height'] / fixed_aia_scale['y']).value

    for region in regions:
        # Next region
        R = regions[region]

        # The mapcube images a piece of Sun that inevitably moves.  Defining
        # the subcube using HPC co-ords will not work in this case.  Therefore
        # we need to convert the HPC co-ords into array indices and return
        # numpy arrays.  Create the subcube
        range_x = [R['llxy_pixel'][1], R['llxy_pixel'][1] + R['height_pixel']]
        range_y = [R['llxy_pixel'][0], R['llxy_pixel'][0] + R['width_pixel']]
        subdata = (mc.as_array())[range_x[0]: range_x[1], range_y[0]: range_y[1], :]

        print('\nSub datacube (size %i, %i, %i)' % subdata.shape)

        # Create a SunPy map that describes the location of the data
        region_submap = mc_layer.submap(range_x * u.pix, range_y * u.pix)

        # Region identifier name
        region_id = sd.ident + '_' + region

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, sd.wave, region]

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube.pkl')

        # Open the file and write out the data we need for step 2
        outputfile = open(ofilename, 'wb')

        # Write out the cube of data we want
        pickle.dump(subdata, outputfile)

        # Write out the times when each layer was taken
        pickle.dump(times, outputfile)

        # Write out the layer at which the rotation etc refer too
        pickle.dump(mc_layer, outputfile)

        # Write out the region information
        pickle.dump(R, outputfile)

        # Write out the regional submap
        pickle.dump(region_submap, outputfile)

        # Close the data
        outputfile.close()
        print('Saved to ' + ofilename)

        # Write out a FITS file
        # hdu = fits.PrimaryHDU(subdata)
        # hdulist = fits.HDUList([hdu])
        # hdulist.writeto(ofilename + '.fits')

#
# Save the regions
#

#
# Plot where the regions are
#
filepath = os.path.join(sd.save_locations['image'], sd.ident + '.regions')
for region in regions.keys():
    filepath = filepath + '.' + region
filepath = filepath + '.png'
step1_plots.plot_regions(mc_layer, regions, filepath)

for region in regions.keys():
    R = regions[region]
    mc_layer_submap = mc_layer.submap(R['xrange'] * u.arcsec, R['yrange'] * u.arcsec)
    step1_plots.plot_regions_hsr2015(mc_layer_submap, filepath + '.submap.png')
