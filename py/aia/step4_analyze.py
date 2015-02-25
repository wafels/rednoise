#
# Step 3 plots.  Load in all the power spectra fit results and make some plots
#
import os
import cPickle as pickle
import study_details as sd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
# Wavelengths we want to analyze
waves = ['171', '193']

# Regions we are interested in
regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']

# Apodization windows
windows = ['hanning']

# Models to fit
model_names = ('power law with constant',)

# Reduced chi-squared limit
rchilimit = [0.9, 1.1]
rchistring = '$\chi^{2}_{r}$'

xlim = [0.5, 2.0]

# Parameters
parameters = ("log(amplitude)", "power law index", "log(background)")


# Create the storage across all models, AIA channels and regions
storage = {}
for model_name in model_names:
    storage[model_name] = {}
    for wave in waves:
        storage[model_name][wave] = {}
        for region in regions:
            storage[model_name][wave][region] = []

def result_filter(result, rchilimit):
    success = result[0]
    rchi2_gt_low_limit = result[1] > rchilimit[0]
    rchi2_lt_high_limit = result[1] < rchilimit[1]
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
                index = np.zeros_like(rchi2)
                success = np.zeros_like(rchi2)

                for i in range(0, nx):
                    for j in range(0, ny):
                        # Get the reduced chi-squared
                        rchi2[j, i] = data[j][i][2]
                        # Get the results
                        result = data[j][i][1]
                        index[j, i] = result['x'][1]
                        success[j, i] = result['success']
        storage[model_name][wave][region] = [success, rchi2, index]


#
# Histograms of model parameters
#
for iwave, wave in enumerate(waves):
    for iparameter, parameter in enumerate(parameters):
        f , axarr = plt.subplots(len(regions), 1, sharex=True)
        for iregion, region in enumerate(regions):
            result = storage[model_name][wave][region]
            mask = result_filter(result, rchilimit)
            index = result[iparameter]
            m = ma.array(index, mask=np.logical_not(mask)).flatten()
            axarr[iregion].hist(m, bins=50, alpha=0.5, label='%s [%i%%]' % (region, np.int(100 * np.sum(mask) / np.float(index.size))), normed=True)
            axarr[iregion].legend()
            axarr[iregion].set_ylabel('pdf')
        axarr[0].set_title('%s, %s' % (wave, model_name))
        axarr[len(regions) - 1].set_xlabel(parameter)
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
        mask1 = result_filter(result1, rchilimit)
        index1 = result1[2]
        for j in range(i+1, len(waves)):
            wave2 = waves[j]
            result2 = storage[model_name][wave2][region]
            mask2 = result_filter(result2, rchilimit)
            index2 = result2[2]

            m1 = ma.array(index1, mask=np.logical_not(mask1 * mask2)).flatten()
            m2 = ma.array(index2, mask=np.logical_not(mask1 * mask2)).flatten()

            # Plot all the results
            plt.scatter(result1, result2, 'bo', label='all')

            # Plot the results with "good" reduced chi-squared
            plt.scatter(m1, m2, 'yo', label='%f<%s<f' % (rchilimit[0], rchistring, rchilimit[1]))
            plt.xlabel(wave1)
            plt.ylabel(wave2)
            plt.xlim(xlim[0], xlim[1])
            plt.plot(xlim, xlim)
            plt.title(region)
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