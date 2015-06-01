#
# Analysis - AIC and BIC.  Plot the spatial distributions of
# the AIC and the BIC
#
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import analysis_details as ad
import analysis_get_data
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
model_names = ('power law with constant and lognormal', 'power law with constant')

# Look at those results that have chi-squared values that give rise to
# probabilities within these values
pvalue = ad.pvalue

# Model fit parameter names
parameters = ad.parameters(model_names)

# Other time-series parameter names
othernames = ad.othernames

# Are the above parameters comparable to values found in ireland et al 2015?
comparable = ad.comparable(model_names)

# Conversion factors to convert the stored parameter values to ones which are
# simpler to understand when plotting them out
conversion = ad.conversion(model_names)

# Informative plot labels
plotname = ad.plotname(model_names)

# Number of parameters we are considering
nparameters = ad.nparameters(model_names)

# Ireland et al 2015 Label
i2015label = i2015.label

#
rchi2limit = ad.rchi2limit(pvalue, nposfreq, nparameters)

# Get the data
storage = analysis_get_data.get_all_data()

# Get the good fit masks
masks = analysis_get_data.get_masks(storage, rchi2limit)


# Plot spatial distributions of the AIC and BIC.  The AIC and BIC for each
# model are subtracted, and the model with the lowest AIC or BIC is preferred.
ic = {"AIC": {}, "BIC": {}}
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

        # Get all the AIC and BIC information
        for imodel_name, model_name in enumerate(model_names):
            z = storage[wave][region][model_name]
            for measure in ("AIC", "BIC"):
                ic[measure][model_name] = analysis_get_data.as_numpy_array(z, ad.ic[measure])

        for measure in ("AIC", "BIC"):
            success0 = analysis_get_data.as_numpy_array(storage[wave][region][model_names[0]], ad.success)
            success1 = analysis_get_data.as_numpy_array(storage[wave][region][model_names[1]], ad.success)
            mask_both_models = success0 * success1

            map_data = ma.array(ic[measure][model_names[0]] - ic[measure][model_names[1]], mask=np.logical_not(mask_both_models))

            my_map = analysis_get_data.make_map(output, region_id, wave, map_data)

            # Make a spatial distribution map that also shows where the bad
            # fits are
            plt.close('all')
            # Set up the palette we will use
            palette = cm.gnuplot2
            # Bad values are those that are masked out
            palette.set_bad('0.75')

            # Begin the plot
            fig, ax = plt.subplots()
            # Plot the map
            ret = my_map.plot(cmap=palette, axes=ax, interpolation='none')

            cbar = fig.colorbar(ret, extend='both', orientation='horizontal',
                                shrink=0.8, label='$\delta$%s (%s-%s)' % (measure, model_names[0], model_names[1]))

            # Fit everything in.
            ax.autoscale_view()

            # Dump to file
            filepath = os.path.join(image, 'spatial_distrib.' + region_id + '.%s.png' % measure)
            print('Saving to ' + filepath)
            plt.savefig(filepath)
