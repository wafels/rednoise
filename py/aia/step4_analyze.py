#
# Step 3 plots.  Load in all the power spectra fit results and make some plots
#
import os
import cPickle as pickle
import study_details as sd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import lnlike_model_fit
import ireland2015_details as i2015
# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Models to fit
model_names = ('power law with constant',)

# Look at those results that have chi-squared values that give rise to
# probabilities within these values
pvalue = np.array([0.025, 0.975])


# Parameters
parameters = ("amplitude", "power law index", "background")
comparable = (False, True, False)
conversion = (1.0 / np.log(10.0), 1.0, 1.0 / np.log(10.0))
plotname = ('$\log_{10}$(amplitude)', "power law index", "$\log_{10}$background")
nparameters = len(parameters)

nposfreq= 899
# Calculate reduced chi-squared properties
rchi2limit = [lnlike_model_fit.rchi2_given_prob(pvalue[1], 1.0, nposfreq - nparameters - 1),
              lnlike_model_fit.rchi2_given_prob(pvalue[0], 1.0, nposfreq - nparameters - 1)]
rchi2s = '$\chi^{2}_{r}$'
rchi2string = '$<$' + rchi2s + '$<$'
rchi2label = '%1.2f%s%1.2f' % (rchi2limit[0], rchi2string, rchi2limit[1])
pstring = '%2.1f%%<p<%2.1f%%' % (100 * pvalue[0], 100 * pvalue[1])
rchi2limitcolor = ['r', 'y']

# Label
i2015label = i2015.label


# Create the storage across all models, AIA channels and regions
storage = {}
for model_name in model_names:
    storage[model_name] = {}
    for wave in waves:
        storage[model_name][wave] = {}
        for region in regions:
            storage[model_name][wave][region] = []

def result_filter(success, rchi2, rchilimit):
    rchi2_gt_low_limit = rchi2[1] > rchilimit[0]
    rchi2_lt_high_limit = rchi2[1] < rchilimit[1]
    return success * rchi2_gt_low_limit * rchi2_lt_high_limit


#
# Main analysis loops
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
# Histograms of reduced chi-squared
#
for iwave, wave in enumerate(waves):
    plt.close('all')
    f, axarr = plt.subplots(len(regions), 1, sharex=True)

    for iregion, region in enumerate(regions):

        # Get the reduced chi-squared
        rchi2 = storage[model_name][wave][region][1]

        # Plot the histogram
        axarr[iregion].hist(rchi2.flatten(), bins=50, alpha=0.5, label='all %s' % region, normed=True)

        # Show legend and define the lines that are plotted
        if iregion == 0:
            axarr[0].axvline(rchi2limit[0], color=rchi2limitcolor[0], linewidth=2, label='expected %s (p$<$%2.1f%%)' % (rchi2s, 100 * pvalue[0]))
            axarr[0].axvline(rchi2limit[1], color=rchi2limitcolor[1], linewidth=2, label='expected %s (p$>$%2.1f%%)' % (rchi2s, 100 - 100 * pvalue[1]))
            axarr[0].legend(framealpha=0.5)
        else:
            # Show legend first, then plot lines - no need for them to appear in the legend
            axarr[iregion].legend(framealpha=0.5)
            axarr[iregion].axvline(rchi2limit[0], color=rchi2limitcolor[0], linewidth=2, label='expected %s (p$<$%2.1f%%)' % (rchi2s, 100 * pvalue[0]))
            axarr[iregion].axvline(rchi2limit[1], color=rchi2limitcolor[1], linewidth=2, label='expected %s (p$>$%2.1f%%)' % (rchi2s, 100 - 100 * pvalue[1]))


        # Get the y limits of the plot and the x limits
        ylim = axarr[iregion].get_ylim()
        axarr[iregion].set_xlim(0.0, 4.0)

        # add in percentage of reduced chi-squared values above and below the upper and lower limits
        actual_pvalue = np.array([np.sum(rchi2 < rchi2limit[0]),
                                        np.sum(rchi2 > rchi2limit[1])]) / np.float64(np.size(rchi2))
        axarr[iregion].text(rchi2limit[0], ylim[0] + 0.67 * (ylim[1] - ylim[0]),
                            '%2.1f%%' % (100 * actual_pvalue[0]),
                            bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
        axarr[iregion].text(rchi2limit[1], ylim[0] + 0.33 * (ylim[1] - ylim[0]),
                            '%2.1f%%' % (100 * actual_pvalue[1]),
                            bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))

        # y axis label
        axarr[iregion].set_ylabel('pdf')


    axarr[0].set_title('observed %s PDFs (%s, model is "%s)"' % (rchi2s, wave, model_name))
    axarr[len(regions) - 1].set_xlabel(rchi2s)
    plt.show()
"""
#
# Maps of reduced chi-squared
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


"""
#
# Histograms of model parameters
#
for iwave, wave in enumerate(waves):
    for iparameter, parameter in enumerate(parameters):
        plt.close('all')
        f, axarr = plt.subplots(len(regions), 1, sharex=True)
        for iregion, region in enumerate(regions):
            result = storage[model_name][wave][region]

            # Generate a mask to filter the results
            mask = result_filter(result[0], result[1], rchi2limit)

            # Parameter to look at
            par = result[2][:, :, iparameter] * conversion[iparameter]

            # Masked array
            m = ma.array(par, mask=np.logical_not(mask)).flatten()

            # Plot all the results
            axarr[iregion].hist(par.flatten(), bins=50, alpha=0.5, label='all %s' % region, normed=True)

            # Plot the best results
            axarr[iregion].hist(m.compressed(), bins=50, alpha=0.5, label='best %s [%i%%]' % (region, np.int(100 * np.sum(mask) / np.float(par.size))), normed=True)

            # Plot the Ireland et al 2015 results
            if comparable[iparameter]:
                z = i2015.df.loc[region]
                i2015value = z[z['waveband'] == int(wave)][parameter]
                axarr[iregion].axvline(i2015value[0], color='r', linewidth=2, label=i2015label)

            # Annotate the plot
            axarr[iregion].legend(framealpha=0.5)
            axarr[iregion].set_ylabel('pdf')
        axarr[0].set_title('%s, model is "%s"' % (wave, model_name))
        axarr[len(regions) - 1].set_xlabel(plotname[iparameter])
        plt.show()
#
# Scatter plots of the power law indices in different channels
#
# Get the limits of the parameters so that they are all plotted on the same
# scale.
for iregion, region in enumerate(regions):
    for i in range(0, len(waves)):
        pass

for iparameter, parameter in enumerate(parameters):

    for iregion, region in enumerate(regions):
        for i in range(0, len(waves)):
            wave1 = waves[i]
            result1 = storage[model_name][wave1][region]
            mask1 = result_filter(result1[0], result1[1], rchi2limit)

            # Parameter to look at
            par1 = result1[2][:, :, iparameter] * conversion[iparameter]

            plt.close('all')
            for j in range(i+1, len(waves)):
                wave2 = waves[j]
                result2 = storage[model_name][wave2][region]
                mask2 = result_filter(result2[0], result2[1], rchi2limit)
                par2 = result2[2][:, :, iparameter] * conversion[iparameter]

                mask = mask1 * mask2

                m1 = ma.array(par1, mask=np.logical_not(mask)).flatten()
                m2 = ma.array(par2, mask=np.logical_not(mask)).flatten()

                # Plot all the results
                plt.scatter(par1.flatten(), par2.flatten(), marker='o', color='b', label='all')

                # Plot the results with "good" reduced chi-squared
                plt.scatter(m1.compressed(), m2.compressed(), marker='o', color='y', label='best [%i%%]' % np.int(100 * np.sum(mask) / np.float(par1.size)) )
                plt.xlabel(wave1 + ' (%s)' % plotname[iparameter])
                plt.ylabel(wave2 + ' (%s)' % plotname[iparameter])
                xlim = [np.min(par1), np.max(par1)]
                ylim = [np.min(par2), np.max(par2)]
                plt.xlim(xlim[0], xlim[1])
                plt.ylim(ylim[0], ylim[1])
                plt.plot([-100, 100], [-100, 100], color='k', label='%s $\equiv$ %s ' % (wave1, wave2))
                plt.title(region + ' (%s, %s)' % (pstring, rchi2label))
                plt.legend(framealpha=0.5)
                plt.show()

#
# Spatial distribution of the power law index.
#
"""