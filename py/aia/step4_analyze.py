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
pvalue = [0.32, 0.68]


# Calculate reduced chi-squared properties
rchilimit = [lnlike_model_fit.rhi2_given_prob(pvalue[1], 1,0, nposfreq - nparameters - 1),
                lnlike_model_fit.rhi2_given_prob(pvalue[0], 1,0, nposfreq - nparameters - 1)]
rchistring = '$\chi^{2}_{r}$'


# Parameters
parameters = ("amplitude", "power law index", "background")
comparable = (False, True, False)
conversion = (1.0 / np.log(10.0), 1.0, 1.0 / np.log(10.0))
plotname = ('$\log_{10}$(amplitude)', "power law index", "$\log_{10}$background")


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
# Histograms of model parameters
#
for iwave, wave in enumerate(waves):
    for iparameter, parameter in enumerate(parameters):
        plt.close('all')
        f, axarr = plt.subplots(len(regions), 1, sharex=True)
        for iregion, region in enumerate(regions):
            result = storage[model_name][wave][region]

            # Generate a mask to filter the results
            mask = result_filter(result[0], result[1], rchilimit)

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


for iregion, region in enumerate(regions):
    for i in range(0, len(waves)):
        wave1 = waves[i]
        result1 = storage[model_name][wave1][region]
        mask1 = result_filter(result1[0], result1[1], rchilimit)
        index1 = result1[2]
        for j in range(i+1, len(waves)):
            wave2 = waves[j]
            result2 = storage[model_name][wave2][region]
            mask2 = result_filter(result2[0], result2[1], rchilimit)
            index2 = result2[2]

            m1 = ma.array(index1, mask=np.logical_not(mask1 * mask2)).flatten()
            m2 = ma.array(index2, mask=np.logical_not(mask1 * mask2)).flatten()

            # Plot all the results
            plt.scatter(index1.flatten(), index2.flatten(), marker='o', color='b', label='all')

            # Plot the results with "good" reduced chi-squared
            plt.scatter(m1.compressed(), m2.compressed(), marker='o', color='y', label='%f<%s<%f' % (rchilimit[0], rchistring, rchilimit[1]))
            plt.xlabel(wave1)
            plt.ylabel(wave2)
            xlim = [np.min(index1), np.max(index1)]
            ylim = [np.min(index2), np.max(index2)]
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])
            plt.plot([0, 8], [0, 8], color='k', label='%s parameter = %s parameter' % (wave1, wave2))
            plt.title(region)
            plt.legend()
            plt.show()

#
# Distributions of reduced chi-squared as a function of region, channel and
# model
#
for iregion, region in enumerate(regions):
    for iwave, wave in enumerate(waves):
        for imodel, model_name in enumerate(model_names):
            pass

#
# Spatial distribution of the power law index.
#
