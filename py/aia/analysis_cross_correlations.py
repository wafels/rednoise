#
# Analysis: perform cross-correlation analyses of the various characterizations
# of the time-series
#
import os
import cPickle as pickle
import study_details as sd
import numpy as np
import numpy.ma as ma
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import ireland2015_details as i2015
import analysis_details
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

# Probability string that corresponds to the reduced chi-squared values
pstring = analysis_details.percentstring(pvalue)

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
# Load in the other characterizations of the time-series
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
# Cross-correlations within channel and region for a given model name
#
# 1. Select and fix model
# 2. for all channels
# 3.     for all regions
# 4.        cross-correlate all the model parameters and other time-series
#           characterizations with each other
#

# Create the cross-correlation storage
cc_within_channel= {}
for wave in waves:
    cc_within_channel[wave] = {}
    for region in regions:
        cc_within_channel[wave][region] = {}
        for parameter1 in all_parameter_names:
            cc_within_channel[wave][region][parameter1] = {}
            for parameter2 in all_parameter_names:
                cc_within_channel[wave][region][parameter1][parameter2] = []

plottype = 'cc.within'
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

        # Get the results
        result = storage[model_name][wave][region]

        # Get the reduced chi-squared
        rchi2 = result[1]

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

        # Get the first parameter
        for iparameter1, parameter1 in enumerate(all_parameter_list):

            # Parameter name
            parameter1_name = all_parameter_names[iparameter1]

            # Data for this parameter, with the mask taken into account
            v1 = conversion[parameter1_name] * ma.array(parameter1, mask=np.logical_not(mask)).compressed()

            # Get the second parameter
            for iparameter2 in range(iparameter1 + 1, len(all_parameter_list)):

                # Get the parameter values
                parameter2 = all_parameter_list[iparameter2]

                # Parameter name
                parameter2_name = all_parameter_names[iparameter2]

                # Data for this parameter, with the mask taken into account
                v2 = conversion[parameter2_name] * ma.array(parameter2, mask=np.logical_not(mask)).compressed()

                # Calculate and store the cross-correlation coefficients
                r = [spearmanr(v1, v2), pearsonr(v1, v2)]
                cc_within_channel[wave][region][parameter1_name][parameter2_name] = r

                # Form the rank correlation string
                rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                # Identifier of the plot
                plotident = rstring + '.' + wave + '.' + region + '.' + parameter1_name + '.' + parameter2_name

                # Make a scatter plot
                title = wave + '-' + region + '(#pixels=%i, used=%3.1f%%)' % (nmask, 100 * nmask/ np.float64(mask.size))
                plt.close('all')
                plt.title(title)
                plt.xlabel(plotname[parameter1_name])
                plt.ylabel(plotname[parameter2_name])
                plt.scatter(v1, v2)
                x0 = plt.xlim()[0]
                ylim = plt.ylim()
                y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                ofilename = plottype + '.scatter.' + plotident + '.png'
                plt.tight_layout()
                plt.savefig(os.path.join(image, ofilename))

                # Make a 2d histogram
                plt.close('all')
                plt.title(title)
                plt.xlabel(plotname[parameter1_name])
                plt.ylabel(plotname[parameter2_name])
                plt.hist2d(v1, v2, bins=40)
                x0 = plt.xlim()[0]
                ylim = plt.ylim()
                y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                ofilename = plottype + '.hist2d.' + plotident + '.png'
                plt.tight_layout()
                plt.savefig(os.path.join(image, ofilename))


#
# Cross-correlate across channel for fixed region and given model name
#
# 1. Select and fix model
# 2. for all regions
# 3.     for channel1 in channels
# 4.         parameter1: get a parameter from the set of model parameters and other time-series characterizations
# 5.         for channel2 in channels
# 6.             parameter2: get a parameter from the set of model parameters and other time-series characterizations
# 7.             cross-correlate parameter1 with parameter2
#

plottype = 'cc.across'
for iregion, region in enumerate(regions):

    # Get the first wave and create a parameter list
    for iwave1, wave1 in enumerate(waves):

        # branch location
        b = [sd.corename, sd.sunlocation, sd.fits_level, wave1, region]

        # Region identifier name
        region_id = sd.datalocationtools.ident_creator(b)

        # Output location
        output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
        image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

        # Output filename
        ofilename = os.path.join(output, region_id + '.datacube')

        # Get the results
        result = storage[model_name][wave1][region]

        # Get the reduced chi-squared
        rchi2 = result[1]

        # Generate a mask to filter the results
        mask1 = analysis_details.result_filter(result[0], result[1], rchi2limit)

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
        all_parameter_list1 = [result[2][:, :, 0], result[2][:, :, 1], result[2][:, :, 2],
                               dtotal, dmax, dmin, dsd, dlnsd]

        for iwave2 in range(iwave1 + 1, len(waves)):

            wave2 = waves[iwave2]

            # branch location
            b = [sd.corename, sd.sunlocation, sd.fits_level, wave2, region]

            # Region identifier name
            region_id = sd.datalocationtools.ident_creator(b)

            # Output location
            output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]
            image = sd.datalocationtools.save_location_calculator(sd.roots, b)["image"]

            # Output filename
            ofilename = os.path.join(output, region_id + '.datacube')

            # Get the results
            result = storage[model_name][wave2][region]

            # Get the reduced chi-squared
            rchi2 = result[1]

            # Generate a mask to filter the results
            mask2 = analysis_details.result_filter(result[0], result[1], rchi2limit)

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
            all_parameter_list2 = [result[2][:, :, 0], result[2][:, :, 1], result[2][:, :, 2],
                                   dtotal, dmax, dmin, dsd, dlnsd]

            # Final mask
            mask = mask1 * mask2
            nmask = np.sum(mask)
            pixels_used_string = '(#pixels=%i, used=%3.1f%%)' % (nmask, 100 * nmask/ np.float64(mask.size))

            for iparameter1, parameter1 in enumerate(all_parameter_list1):

                # Parameter name
                parameter1_name = all_parameter_names[iparameter1]

                # Data for this parameter, with the mask taken into account
                v1 = conversion[parameter1_name] * ma.array(parameter1, mask=np.logical_not(mask)).compressed()

                for iparameter2, parameter2 in enumerate(all_parameter_list2):

                    # Parameter name
                    parameter2_name = all_parameter_names[iparameter2]

                    # Data for this parameter, with the mask taken into account
                    v2 = conversion[parameter2_name] * ma.array(parameter2, mask=np.logical_not(mask)).compressed()

                    # Calculate and store the cross-correlation coefficients
                    r = [spearmanr(v1, v2), pearsonr(v1, v2)]
                    #cc_within_channel[wave][region][parameter1_name][parameter2_name] = r

                    # Form the rank correlation string
                    rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                    # Identifier of the plot
                    plotident = rstring + '.' + region + '.' + wave1 + '.' + parameter1_name + '.' +  wave2 + '.' + parameter2_name

                    # Make a scatter plot
                    title = region + '-' + wave1 + '(' + plotname[parameter1_name] + ')' + wave2 + '(' + plotname[parameter2_name] + ')' + pixels_used_string
                    xlabel = wave1 + ' ' + plotname[parameter1_name]
                    ylabel = wave2 + ' ' + plotname[parameter2_name]
                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.scatter(v1, v2)
                    xlim = plt.xlim()
                    x0 = xlim[0]
                    ylim = plt.ylim()
                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                    plt.text(x0, y0, 'Pearson=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                    plt.text(x0, y1, 'Spearman=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                    # If the same type of parameters are being plotted against
                    # each other, plot an equality line and include a legend
                    if parameter1_name == parameter2_name:
                        parameter_extent = [np.min([xlim, ylim]), np.max([xlim, ylim])]
                        plt.plot(parameter_extent, parameter_extent, label='%s=%s' % (xlabel, ylabel), color='r', linewidth=4)
                        plt.legend(framealpha=0.5)
                    ofilename = plottype + '.scatter.' + plotident + '.png'
                    plt.tight_layout()
                    plt.savefig(os.path.join(image, ofilename))

                    # Make a 2d histogram
                    plt.close('all')
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.hist2d(v1, v2, bins=40)
                    xlim = plt.xlim()
                    x0 = xlim[0]
                    ylim = plt.ylim()
                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                    plt.text(x0, y0, 'Pearson=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                    plt.text(x0, y1, 'Spearman=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                    # If the same type of parameters are being plotted against
                    # each other, plot an equality line and include a legend
                    if parameter1_name == parameter2_name:
                        parameter_extent = [np.min([xlim, ylim]), np.max([xlim, ylim])]
                        plt.plot(parameter_extent, parameter_extent, label='%s=%s' % (xlabel, ylabel), color='r', linewidth=4)
                        plt.legend(framealpha=0.5)
                    ofilename = plottype + '.hist2d.' + plotident + '.png'
                    plt.tight_layout()
                    plt.savefig(os.path.join(image, ofilename))
