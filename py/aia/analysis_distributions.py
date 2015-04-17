#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
import cPickle as pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
import astropy.units as u
import sunpy.map
import sunpy.net.hek as hek
from sunpy.physics.transforms.solar_rotation import rot_hpc
import analysis_details
import ireland2015_details as i2015
import study_details as sd

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Number of positive frequencies in the power spectra
nposfreq = 899

# Model results to examine
model_names = ('power law with constant',)

# Look at those results that have chi-squared values that give rise to
# probabilities within these values
pvalue = analysis_details.pvalue

# Model fit parameter names
parameters = analysis_details.parameters(model_names)

# Other time-series parameter names
othernames = analysis_details.othernames

# Are the above parameters comparable to values found in ireland et al 2015?
comparable = analysis_details.comparable(model_names)

# Conversion factors to convert the stored parameter values to ones which are
# simpler to understand when plotting them out
conversion = analysis_details.conversion(model_names)

# Informative plot labels
plotname = analysis_details.plotname(model_names)

# Number of parameters we are considering
nparameters = len(parameters)

# Calculate reduced chi-squared limits for a given range of pvalues
rchi2limit = analysis_details.rchi2limit(pvalue, nposfreq, nparameters)

# Create a label which shows the limits of the reduced chi-squared value. Also
# define some colors that signify the upper and lower levels of the reduced
# chi-squared
rchi2label = analysis_details.rchi2label(rchi2limit)
rchi2limitcolor = analysis_details.rchi2limitcolor
rchi2s = analysis_details.rchi2s

# Probability string that corresponds to the reduced chi-squared values
pstring = analysis_details.percentstring(pvalue)
percent_lo = analysis_details.percent_lo(pvalue)
percent_hi = analysis_details.percent_hi(pvalue)

# Ireland et al 2015 Label
i2015label = i2015.label


# Create the storage across all models, AIA channels and regions
storage = {}
for model_name in model_names:
    storage[model_name] = {}
    for wave in waves:
        storage[model_name][wave] = {}
        for region in regions:
            storage[model_name][wave][region] = []

# Storage for the parameter limits
param_range = {}
for wave in waves:
    param_range[wave] = {}
    for region in regions:
        param_range[wave][region] = {}


#
# Load in the fit results
#
for iwave, wave in enumerate(waves):

    for iregion, region in enumerate(regions):

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

        # Region identifier name
        region_id = sd.datalocationtools.ident_creator(b)

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

        # Go through all the windows
        for iwindow, window in enumerate(windows):

            # Output filename
            ofilename = os.path.join(output, region_id + '.datacube.' + window)

            # General notification that we have a new data-set
            print('Loading New Data')
            # Which wavelength?
            print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
            # Which region
            print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Load in the fit results
            filepath = os.path.join(output, ofilename + '.lnlike_fit_results.pkl')
            print('Loading results to ' + filepath)
            f = open(filepath, 'rb')
            results = pickle.load(f)
            f.close()

            # Load in the emission results

            for itm, model_name in enumerate(model_names):

                data = results[model_name]

                ny = len(data)
                nx = len(data[0])

                rchi2 = np.zeros((ny, nx))
                parameter_values = np.zeros((ny, nx, 3))
                success = np.zeros_like(rchi2)

                for i in range(0, nx):
                    for j in range(0, ny):
                        # Get the reduced chi-squared
                        rchi2[j, i] = data[j][i][2]
                        # Get the results
                        result = data[j][i][1]
                        parameter_values[j, i, :] = result['x']
                        success[j, i] = result['success']
        storage[model_name][wave][region] = [success, rchi2, parameter_values]

#
# Create the all parameter list of data and their names
#
all_parameter_names = []
for name in parameters:
    all_parameter_names.append(name)
for name in othernames:
    all_parameter_names.append(name)

#
# Load in the other characterizations of the time-series and plot
#
for iwave, wave in enumerate(waves):

    # Get parameter we want to plot
    for iparameter1, parameter1 in enumerate(all_parameter_names):

        # Parameter name
        parameter1_name = all_parameter_names[iparameter1]

        # All the regions appear in one plot
        plt.close('all')
        f, axarr = plt.subplots(len(regions), 1, sharex=True)
        f.set_size_inches(8.0, 16.0)

        # Go through all the regions
        for iregion, region in enumerate(regions):

            # branch location
            b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

            # Region identifier name
            region_id = sd.datalocationtools.ident_creator(b)

            # Output location
            output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
            image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

            # Output filename
            ofilename = os.path.join(output, region_id + '.datacube')

            # Get the results
            result = storage[model_name][wave][region]

            # Generate a mask to filter the results
            mask = analysis_details.result_filter(result[0], result[1], rchi2limit)
            nmask = np.sum(mask)
            pixels_used_string = '(#pixels=%i, used=%3.1f%%)' % (nmask, 100 * nmask/ np.float64(mask.size))

            # Number of results
            nmask = np.sum(mask)

            # Load the other time-series parameters
            ofilename = ofilename + '.' + window
            filepath = os.path.join(output, ofilename + '.summary_stats.npz')
            with np.load(filepath) as otsp:
                dtotal = otsp['dtotal']
                dmax = otsp['dmax']
                dmin = otsp['dmin']
                dsd = otsp['dsd']
                dlnsd = otsp['dlnsd']

            # Create a list containing all the parameters
            all_parameter_list = [result[2][:, :, 0], result[2][:, :, 1], result[2][:, :, 2],
                                  dtotal, dmax, dmin, dsd, dlnsd, result[1]]

            # Which parameter are we looking at
            parameter1 = all_parameter_list[iparameter1]

            # Data for this parameter, with the mask taken into account
            v1 = conversion[parameter1_name] * ma.array(parameter1, mask=np.logical_not(mask)).compressed()

            # Parameter limits
            param_range[wave][region][parameter1_name] = [v1.min(), v1.max()]

            # Plot the histogram
            axarr[iregion].hist(v1.flatten(), bins=50, label='good %s' % region, normed=True, alpha=0.5)
            axarr[iregion].hist(conversion[parameter1_name] * parameter1.flatten(), bins=50, label='all %s' % region, normed=True, alpha=0.5)

            # Show legend and define the lines that are plotted
            if iregion == 0:
                axarr[0].legend(framealpha=0.5)
            else:
                # Show legend first, then plot lines - no need for them to
                # appear in the legend
                axarr[iregion].legend(framealpha=0.5)

            # y axis label
            axarr[iregion].set_ylabel('pdf')

        # File name to put the image in the correct
        filepath = os.path.join(os.path.join(os.path.dirname(sd.save_locations['image']), wave), sd.ident + '.observed.%s-%s.pdfs.png' % (wave, parameter1_name))

        # Finish the plot
        axarr[0].set_title('observed %s-%s PDFs' % (wave, plotname[parameter1_name]))
        axarr[len(regions) - 1].set_xlabel(plotname[parameter1_name])
        plt.savefig(os.path.join(filepath))


#
# Spatial distribution of quantities
#
#
# Create the all parameter list of data and their names
#
all_parameter_names = []
for name in parameters:
    all_parameter_names.append(name)
for name in othernames:
    all_parameter_names.append(name)

#
# Define the parameter ranges over wavelength
#
#  Storage for the parameter limits
param_lims = {}
for region in regions:
    param_lims[region] = {}
    for parameter1 in all_parameter_names:
        param_lims[region][parameter1] = []

for iregion, region in enumerate(regions):
    for parameter1 in all_parameter_names:
        for iwave, wave in enumerate(waves):
            lo_lim = param_range[wave][region][parameter1][0]
            hi_lim = param_range[wave][region][parameter1][1]
            if iwave == 0:
                param_lims[region][parameter1] = [param_range[wave][region][parameter1][0],
                                                  param_range[wave][region][parameter1][1]]
            if param_lims[region][parameter1][0] < lo_lim:
                param_lims[region][parameter1][0] = lo_lim
            if param_lims[region][parameter1][1] > hi_lim:
                param_lims[region][parameter1][1] = hi_lim

# -----------------------------------------------------------------------------
# Get the sunspot details at the time of its detection
#
client = hek.HEKClient()
qr = client.query(hek.attrs.Time("2012-09-23 01:00:00", "2012-09-23 02:00:00"), hek.attrs.EventType('SS'))
p1 = qr[0]["hpc_boundcc"][9: -2]
p2 = p1.split(',')
p3 = [v.split(" ") for v in p2]
p4 = np.asarray([(eval(v[0]), eval(v[1])) for v in p3])
polygon = np.zeros([1, len(p2), 2])
polygon[0, :, :] = p4[:, :]


#
# Load in the other characterizations of the time-series and plot
#
for iwave, wave in enumerate(waves):

    # Get parameter we want to plot
    for iparameter1, parameter1 in enumerate(all_parameter_names):

        # Parameter name
        parameter1_name = all_parameter_names[iparameter1]

        # All the regions appear in one plot
        plt.close('all')
        f, axarr = plt.subplots(len(regions), 1)
        f.set_size_inches(8.0, 16.0)

        # Go through all the regions
        for iregion, region in enumerate(regions):

            # branch location
            b = [sd.corename, sd.sunlocation, sd.fits_level, wave, region]

            # Region identifier name
            region_id = sd.datalocationtools.ident_creator(b)

            # Output location
            output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
            image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

            # Output filename
            ofilename = os.path.join(output, region_id + '.datacube')

            # Get the results
            result = storage[model_name][wave][region]

            # Generate a mask to filter the results
            mask = analysis_details.result_filter(result[0], result[1], rchi2limit)
            nmask = np.sum(mask)
            pixels_used_string = '(#pixels=%i, used=%3.1f%%)' % (nmask, 100 * nmask/ np.float64(mask.size))

            # Number of results
            nmask = np.sum(mask)

            # Load the other time-series parameters
            ofilename = ofilename + '.' + window
            filepath = os.path.join(output, ofilename + '.summary_stats.npz')
            with np.load(filepath) as otsp:
                dtotal = otsp['dtotal']
                dmax = otsp['dmax']
                dmin = otsp['dmin']
                dsd = otsp['dsd']
                dlnsd = otsp['dlnsd']

            # Create a list containing all the parameters
            all_parameter_list = [result[2][:, :, 0], result[2][:, :, 1], result[2][:, :, 2],
                                  dtotal, dmax, dmin, dsd, dlnsd, result[1]]

            # Which parameter are we looking at
            parameter1 = all_parameter_list[iparameter1]

            # Data for this parameter, with the mask taken into account
            v1 = conversion[parameter1_name] * ma.array(parameter1, mask=np.logical_not(mask))

            # Get the map: Open the file that stores it and read it in
            map_data_filename = os.path.join(output, region_id + '.datacube.pkl')
            get_map_data = open(map_data_filename, 'rb')
            _dummy = pickle.load(get_map_data)
            _dummy = pickle.load(get_map_data)
            # map layer
            mc_layer = pickle.load(get_map_data)
            # Specific region information
            R = pickle.load(get_map_data)
            get_map_data.close()

            # Make a spatial distribution map that also shows where the bad
            # fits are
            plt.close('all')
            # Set up the palette we will use
            palette = cm.brg
            # Bad values are those that are masked out
            palette.set_bad('k', 1.0)
            im = plt.imshow(v1,
                            extent=(R["xrange"][0], R["xrange"][1], R["yrange"][0], R["yrange"][1]),
                            interpolation='none',
                            cmap=palette,
                            norm=colors.Normalize(vmin=param_lims[region][parameter1_name][0],
                                                  vmax=param_lims[region][parameter1_name][1],
                                                  clip=False),
                            origin='lower')
            plt.title(wave + ': ' + region + ': ' + plotname[parameter1_name])
            plt.xlabel('solar X (arcseconds)')
            plt.ylabel('solar Y (arcseconds)')
            plt.colorbar(im, extend='both', orientation='horizontal',
                         shrink=0.8, label=plotname[parameter1_name])

            # Create the map
            header = {'cdelt1': 0.6, 'cdelt2': 0.6, "crval1": -332.5, "crval2": 17.5,
                  "telescop": 'AIA', "detector": "AIA", "wavelnth": "171",
                  "date-obs": mc_layer.date}
            my_map = sunpy.map.Map(v1, header)
            # Begin the plot
            fig, ax = plt.subplots()
            # Plot the map
            my_map.plot(cmap=palette,
                        interpolation='none',
                        norm=colors.Normalize(vmin=param_lims[region][parameter1_name][0],
                                              vmax=param_lims[region][parameter1_name][1],
                                              clip=False))
            # Looking at sunspots?  If so, overplot the outline of the sunspot,
            # ensuring that it has been rotated to the time of the layer_index
            if region == 'sunspot':
                rotated_polygon = np.zeros_like(polygon)
                for i in range(0, len(p2)):
                    new_coords = rot_hpc(polygon[0, i, 0] * u.arcsec,
                                         polygon[0, i, 1] * u.arcsec,
                                         sunspot_date,
                                         mc_layer.date)
                    rotated_polygon[0, i, 0] = new_coords[0].value
                    rotated_polygon[0, i, 1] = new_coords[1].value
                # Create the collection
                coll = PolyCollection(rotated_polygon,
                                      alpha=1.0,
                                      edgecolors=['k'],
                                      facecolors=['none'],
                                      linewidth=[5])
                # Add to the plot
                ax.add_collection(coll)

            # Fit everything in.
            ax.autoscale_view()

            filepath = os.path.join(image, 'spatial_distrib.' + region_id + '.' + parameter1_name + '.png')
            print('Saving to ' + filepath)
            plt.savefig(filepath)

