#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
import cPickle as pickle
import study_details as sd
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import analysis_details
import ireland2015_details as i2015

# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Number of positive frequencies in the power spectra
nposfreq= 899

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
# Load in the other characterizations of the time-series
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
                                  dtotal, dmax, dmin, dsd, dlnsd]

            # Which parameter are we looking at
            parameter1 = all_parameter_list[iparameter1]

            # Data for this parameter, with the mask taken into account
            v1 = conversion[parameter1_name] * ma.array(parameter1, mask=np.logical_not(mask)).compressed()

            # Plot the histogram
            axarr[iregion].hist(v1.flatten(), bins=50, label='good %s' % region, normed=True, alpha=0.5)
            axarr[iregion].hist(conversion[parameter1_name] * parameter1.flatten(), bins=50, label='all %s' % region, normed=True, alpha=0.5)

            # Show legend and define the lines that are plotted
            if iregion == 0:
                axarr[0].legend(framealpha=0.5)
            else:
                # Show legend first, then plot lines - no need for them to appear in the legend
                axarr[iregion].legend(framealpha=0.5)

            # y axis label
            axarr[iregion].set_ylabel('pdf')

        # File name to put the image in the correct
        filepath = os.path.join(os.path.join(os.path.dirname(sd.save_locations['image']), wave), sd.ident + '.observed.%s-%s.pdfs.png' % (wave, parameter1_name))

        # Finish the plot
        axarr[0].set_title('observed %s-%s PDFs' % (wave, plotname[parameter1_name]))
        axarr[len(regions) - 1].set_xlabel(plotname[parameter1_name])
        plt.savefig(os.path.join(filepath))

"""
#
# Spatial distribution of quantities
#

for iwave, wave in enumerate(waves):
    f, axarr = plt.subplots(len(regions), 1)

    for iregion, region in enumerate(regions):
        rchi2 = storage[model_name][wave][region][1]
        rchi2[rchi2 > rchi2limit[1]] = rchi2limit[1]
        rchi2[rchi2 < rchi2limit[0]] = rchi2limit[0]
        cax = axarr[iregion].imshow(rchi2)
        axarr[iregion].set_title('%s, model is "%s" in %s' % (wave, model_name, region))
        f.colorbar(cax, ax=axarr[iregion])
    plt.show()
"""