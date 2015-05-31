#
# Analysis - AIC and BIC.  Plot the spatial distributions of
# the AIC and the BIC
#
import os
import analysis_details
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

#
rchi2limit = analysis_details.rchi2limit(pvalue, nposfreq, nparameters)

# Get the data
storage = analysis_get_data.get_all_data()

# Get the good fit masks
masks = analysis_get_data.get_masks(storage, rchi2limit)


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
            z = storage[wave][region][model_name]
            aic[imodel_name] = analysis_get_data.as_numpy_array(z, [])

