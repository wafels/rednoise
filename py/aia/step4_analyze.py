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
regions = ['sunspot', 'loop footpoints', 'quiet Sun', 'moss']

# Apodization windows
windows = ['hanning']

# Models to fit
model_names = ('power law with constant',)

#
rchilimit = [0.5, 2.0]

# Storage across all models, AIA channels and regions
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
    # Histograms of indices
    #
    for iregion, region in enumerate(regions):
        result = storage[model_name][wave][region]
        mask = result_filter(result, rchilimit)
        index = result[2]
        m = ma.array(index, mask=np.logical_not(mask)).flatten()
        plt.hist(m, bins=50, alpha=0.5, label='%s [%i%%]' % (region, np.int(100 * np.sum(mask) / np.float(index.size))), normed=True)
    plt.xlim(0.5, 4.0)
    plt.legend()
    plt.xlabel('power law index')
    plt.ylabel('probability density function')
    plt.title(wave)
    plt.show()

"""
#
# Scatter plots
#
for iregion, region in enumerate(regions):
    for wave1 in waves:
        result1 = storage[model_name][wave1][region]
        mask1 = result_filter(result1, rchilimit)
        index1 = result1[2]
        for wave2 in waves:
            result2 = storage[model_name][wave2][region]
            mask2 = result_filter(result2, rchilimit)
            index2 = result2[2]

            m1 = ma.array(index1, mask=np.logical_not(mask1 * mask2)).flatten()
            m2 = ma.array(index2, mask=np.logical_not(mask1 * mask2)).flatten()

            plt.scatter(m1, m2)
            plt.xlabel(wave1)
            plt.ylabel(wave2)
            plt.title(region)
            plt.show()
"""