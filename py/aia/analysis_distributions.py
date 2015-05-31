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


def summary_statistics(a):
    return {"mean": np.mean(a),
            "median": np.percentile(a, 50.0),
            "lo": np.percentile(a, 2.5),
            "hi": np.percentile(a, 97.5),
            "min":np.min(a),
            "max": np.max(a),
            "std": np.std(a)}


# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Number of positive frequencies in the power spectra
nposfreq = 899

# Model results to examine
model_names = ('power law with constant and lognormal', 'power law with constant')

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
nparameters = analysis_details.nparameters(model_names)

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
                parameter_values = np.zeros((ny, nx, nparameters[model_name]))
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
all_parameter_names = {}
for model_name in model_names:
    all_parameter_names[model_name] = []
    for name in parameters[model_name]:
        all_parameter_names[model_name].append(name)
    for name in othernames:
        all_parameter_names[model_name].append(name)

#
# Load in the other characterizations of the time-series and plot
#
for iwave, wave in enumerate(waves):

    # Model name
    for itm, model_name in enumerate(model_names):

        # Number of parameters
        npars = nparameters[model_name]

        # The parameters of this model
        these_parameter_names = all_parameter_names[model_name]

        # Calculate reduced chi-squared limits for a given range of
        # pvalues
        rchi2limit = analysis_details.rchi2limit(pvalue, nposfreq, npars)

        # Create a label which shows the limits of the reduced
        # chi-squared value. Also define some colors that signify the
        # upper and lower levels of the reduced chi-squared
        rchi2label = analysis_details.rchi2label(rchi2limit)
        rchi2limitcolor = analysis_details.rchi2limitcolor
        rchi2s = analysis_details.rchi2s

        # Probability string that corresponds to the reduced
        # chi-squared values
        pstring = analysis_details.percentstring(pvalue)
        percent_lo = analysis_details.percent_lo(pvalue)
        percent_hi = analysis_details.percent_hi(pvalue)

        # Get parameter we want to plot
        for iparameter1, parameter1 in enumerate(these_parameter_names):

            # Parameter name
            parameter1_name = these_parameter_names[iparameter1]

            # All the regions appear in one plot
            plt.close('all')
            f, axarr = plt.subplots(len(regions), 1, sharex=True)
            f.set_size_inches(8.0, 16.0)

            # Summary stats storage
            summary_stats = {}

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
                all_parameter_list = []
                for ipar in range(0, npars):
                    all_parameter_list.append(result[2][:, :, ipar])
                all_parameter_list.append(dtotal)
                all_parameter_list.append(dmax)
                all_parameter_list.append(dmin)
                all_parameter_list.append(dsd)
                all_parameter_list.append(dlnsd)
                all_parameter_list.append(result[1])

                # Which parameter are we looking at
                parameter1 = all_parameter_list[iparameter1]

                # Data for this parameter, with the mask taken into account
                v1 = conversion[parameter1_name] * ma.array(parameter1, mask=np.logical_not(mask)).compressed()

                # Parameter limits
                param_range[wave][region][parameter1_name] = [v1.min(), v1.max()]

                # Plot the histogram
                # Good data
                axarr[iregion].hist(v1.flatten(), bins=50, label='good %s' % region, normed=True, alpha=0.5)
                # All data
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

                # Summary statistics of the distribution - mean, mode, median,
                # etc
                summary_stats[wave] = summary_statistics(v1)
                if parameter1_name == 'power law index' and wave == '171':
                    print region, summary_stats[wave]

            # File name to put the image in the correct
            filepath = os.path.join(os.path.join(os.path.dirname(sd.save_locations['image']), wave), sd.ident + '.observed.%s-%s.pdfs.%s.png' % (wave, parameter1_name, model_name))

            # Finish the plot
            axarr[0].set_title('observed %s-%s PDFs' % (wave, plotname[parameter1_name]))
            axarr[len(regions) - 1].set_xlabel(plotname[parameter1_name])
            print("Saving " + filepath)
            plt.savefig(os.path.join(filepath))

#
# Spatial distribution of quantities
#

#
# Define the parameter ranges over wavelength
#
#  Storage for the parameter limits
param_lims = {}
for model_name in model_names:
    param_lims[model_name] = {}
    for region in regions:
        param_lims[model_name][region] = {}
        for parameter1 in all_parameter_names[model_name]:
            param_lims[model_name][region][parameter1] = []

for model_name in model_names:
    for iregion, region in enumerate(regions):
        for parameter1 in all_parameter_names[model_name]:
            for iwave, wave in enumerate(waves):
                lo_lim = param_range[wave][region][parameter1][0]
                hi_lim = param_range[wave][region][parameter1][1]
                if iwave == 0:
                    param_lims[model_name][region][parameter1] = [param_range[wave][region][parameter1][0],
                                                      param_range[wave][region][parameter1][1]]
                if param_lims[model_name][region][parameter1][0] < lo_lim:
                    param_lims[model_name][region][parameter1][0] = lo_lim
                if param_lims[model_name][region][parameter1][1] > hi_lim:
                    param_lims[model_name][region][parameter1][1] = hi_lim
            # Override
            if parameter1 == 'power law index':
                param_lims[model_name][region][parameter1][0] = 1.0
                param_lims[model_name][region][parameter1][1] = 4.0


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
sunspot_date = qr[0]['event_endtime']

#
# Load in the other characterizations of the time-series and plot
#
for iwave, wave in enumerate(waves):

    # Model name
    for itm, model_name in enumerate(model_names):

        # Number of parameters
        npars = nparameters[model_name]

        # The parameters of this model
        these_parameter_names = all_parameter_names[model_name]

        # Calculate reduced chi-squared limits for a given range of
        # pvalues
        rchi2limit = analysis_details.rchi2limit(pvalue, nposfreq, npars)

        # Create a label which shows the limits of the reduced
        # chi-squared value. Also define some colors that signify the
        # upper and lower levels of the reduced chi-squared
        rchi2label = analysis_details.rchi2label(rchi2limit)
        rchi2limitcolor = analysis_details.rchi2limitcolor
        rchi2s = analysis_details.rchi2s

        # Probability string that corresponds to the reduced
        # chi-squared values
        pstring = analysis_details.percentstring(pvalue)
        percent_lo = analysis_details.percent_lo(pvalue)
        percent_hi = analysis_details.percent_hi(pvalue)

        # Get parameter we want to plot
        for iparameter1, parameter1 in enumerate(these_parameter_names):

            # Parameter name
            parameter1_name = these_parameter_names[iparameter1]

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
                all_parameter_list = []
                for ipar in range(0, nparameters[model_name]):
                    all_parameter_list.append(result[2][:, :, ipar])
                all_parameter_list.append(dtotal)
                all_parameter_list.append(dmax)
                all_parameter_list.append(dmin)
                all_parameter_list.append(dsd)
                all_parameter_list.append(dlnsd)
                all_parameter_list.append(result[1])

                # Which parameter are we looking at
                parameter1 = all_parameter_list[iparameter1]

                # Data for this parameter, with the mask taken into account.
                # The mask is defined such that True implies a "good value", and
                # False implies "bad value".  In numpy masked arrays, "True" implies
                # "masked value", and "False" implies "non-masked value".
                # Therefore, when defining the masked array we need to take the
                # logical not of the mask.
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
                palette.set_bad('0.75')

                # Create the map
                header = {'cdelt1': 0.6, 'cdelt2': 0.6, "crval1": -332.5, "crval2": 17.5,
                      "telescop": 'AIA', "detector": "AIA", "wavelnth": wave,
                      "date-obs": mc_layer.date}
                my_map = sunpy.map.Map(v1, header)
                # Begin the plot
                fig, ax = plt.subplots()
                # Plot the map
                ret = my_map.plot(cmap=palette, axes=ax,
                            interpolation='none',
                            norm=colors.Normalize(vmin=param_lims[model_name][region][parameter1_name][0],
                                                  vmax=param_lims[model_name][region][parameter1_name][1],
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

                cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                    shrink=0.8, label=plotname[parameter1_name] + ' : %s' % model_name)

                # Fit everything in.
                ax.autoscale_view()

                # Dump to file
                filepath = os.path.join(image, 'spatial_distrib.' + region_id + '.' + parameter1_name + '.%s.png' % (model_name))
                print('Saving to ' + filepath)
                plt.savefig(filepath)

#
# Which model has better fitness?  Use the selection masks to make this
# determination.  There are four possible outcomes
#
# -1 = neither model has a good fitness
#  0 = model 0 has a good fitness, model 1 does not
#  1 = model 1 has a good fitness, model 0 does not
#  2 = models 0 and 1 both have a good fitness
#
masks = {}
aic_all = {}
bic_all = {}
rchi2all = {}
for wave in waves:
    masks[wave] = {}
    rchi2all[wave] = {}
    aic_all[wave] = {}
    bic_all[wave] = {}

    for region in regions:
        masks[wave][region] = {}
        rchi2all[wave][region] = {}
        aic_all[wave][region] = {}
        bic_all[wave][region] = {}

        for model_name in model_names:
            masks[wave][region][model_name] = {}
            rchi2all[wave][region][model_name] = {}
            aic_all[wave][region][model_name] = {}
            bic_all[wave][region][model_name] = {}

for iwave, wave in enumerate(waves):

    # Model name
    for itm, model_name in enumerate(model_names):

        # Number of parameters
        npars = nparameters[model_name]

        # The parameters of this model
        these_parameter_names = all_parameter_names[model_name]

        # Calculate reduced chi-squared limits for a given range of
        # pvalues
        rchi2limit = analysis_details.rchi2limit(pvalue, nposfreq, npars)

        # Create a label which shows the limits of the reduced
        # chi-squared value. Also define some colors that signify the
        # upper and lower levels of the reduced chi-squared
        rchi2label = analysis_details.rchi2label(rchi2limit)
        rchi2limitcolor = analysis_details.rchi2limitcolor
        rchi2s = analysis_details.rchi2s

        # Probability string that corresponds to the reduced
        # chi-squared values
        pstring = analysis_details.percentstring(pvalue)
        percent_lo = analysis_details.percent_lo(pvalue)
        percent_hi = analysis_details.percent_hi(pvalue)

        # Go through all the regions
        for iregion, region in enumerate(regions):

            # Get the results
            result = storage[model_name][wave][region]

            # Generate a mask to filter the results
            masks[wave][region][model_name] = analysis_details.result_filter(result[0], result[1], rchi2limit)

            # Get the reduced chi-squared for each of these
            rchi2all[wave][region][model_name] = result[1]

            # Get the AIC and the BIC
            aic_all[wave][region][model_name] = result[1]
            bic_all[wave][region][model_name] = result[1]


# Plot spatial distributions of the model fitness, comparing two models to each
# other.
for iwave, wave in enumerate(waves):
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

        ny = masks[wave][region][model_names[0]].shape[0]
        nx = masks[wave][region][model_names[0]].shape[1]
        this_mask = np.zeros((2, ny, nx))
        final_mask = np.zeros((ny, nx))
        # Get all the mask details
        for imodel_name, model_name in enumerate(model_names):
            this_mask[imodel_name, :, :] = masks[wave][region][model_name]

        # Both models ok
        masking = np.logical_and(this_mask[0, :, :], this_mask[1, :, :])
        final_mask[np.where(masking)] = 2

        # First model fits
        masking = np.logical_and(this_mask[0, :, :], np.logical_not(this_mask[1, :, :]))
        final_mask[np.where(masking)] = 0

        # Second model fits
        masking = np.logical_and(this_mask[1, :, :], np.logical_not(this_mask[0, :, :]))
        final_mask[np.where(masking)] = 1

        # Neither model fits
        masking = np.logical_and(np.logical_not(this_mask[1, :, :]), np.logical_not(this_mask[0, :, :]))
        final_mask[np.where(masking)] = -1

        # Plot
        plt.close('all')
        # Set up the palette we will use
        palette = cm.jet
        # Bad values are those that are masked out
        palette.set_bad('0.75')

        # Create the map
        header = {'cdelt1': 0.6, 'cdelt2': 0.6, "crval1": -332.5, "crval2": 17.5,
              "telescop": 'AIA', "detector": "AIA", "wavelnth": wave,
              "date-obs": mc_layer.date}
        my_map = sunpy.map.Map(final_mask, header)
        # Begin the plot
        fig, ax = plt.subplots()
        # Plot the map
        ret = my_map.plot(cmap=palette, axes=ax,
                    interpolation='none',
                    norm=colors.Normalize())
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

        cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                            shrink=0.8,
                            label='model selection (based on %s)' % rchi2s,
                            ticks=[-1, 0, 1, 2])
        cbar.ax.set_xticklabels(['neither', model_names[0], model_names[1],
                                'both'])
        cbar.ax.tick_params(labelsize=10)

        # Fit everything in.
        ax.autoscale_view()

        # Dump to file
        filepath = os.path.join(image, 'spatial_distrib.model_selection_reduced_chi2.' + region_id + '.png')
        print('Saving to ' + filepath)
        plt.savefig(filepath)





# Plot spatial distributions of the AIC and BIC.  The AIC and BIC for each
# model are subtracted, and the model with the lowest AIC or BIC is preferred.
for iwave, wave in enumerate(waves):
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

        # Get all the mask details
        for imodel_name, model_name in enumerate(model_names):
            this_mask[imodel_name, :, :] = results[]
