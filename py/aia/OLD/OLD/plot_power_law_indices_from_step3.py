#
#
#
import os
#matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
#rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
import pickle
import numpy as np
import datatools

from py.aia.OLD.paper1 import ireland2014


plt.ion()
plt.close('all')
#
# Set up which data to look at
#
dataroot = '~/Data/AIA/'
ldirroot = '~/ts/pickle_cc_False_dr_False/'
sfigroot = '~/ts/img_cc_False_dr_False_2/'
scsvroot = '~/ts/csv_cc_False_dr_False_2/'
corename = 'study2'
sunlocation = 'spoca665'
#sunlocation = 'equatorial'
fits_level = '1.5'
waves = ['171', '193']
regions = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
windows = ['hanning']
manip = 'relative'

freqfactor = [1000, 'mHz']
savefig_format = 'png'


five_min = freqfactor[0] * 1.0 / 300.0
three_min = freqfactor[0] * 1.0 / 180.0

coherence_wsize = 3

solar_radius_in_arcseconds = 978.0

linewidth = 4

sunday_name = {"equatorial": "quiet Sun",
               "spoca665": "SPoCA 665"}

#
# Storage for the power at the center of each region
#
nregions = len(regions)
indices = np.zeros((nregions))
indices68 = np.zeros((nregions, 2))
indices95 = np.zeros((nregions, 2))
radial_distance = {'171': [], '193': []}
fsd = {}

index_results = {}
for iwave, wave in enumerate(waves):
    # Which wavelength?
    print('Wave: ' + wave + ' (%i out of %i)' % (iwave + 1, len(waves)))
    for iregion, region in enumerate(regions):
        # Which region
        print('Region: ' + region + ' (%i out of %i)' % (iregion + 1, len(regions)))

        # Now that the loading and saving locations are seot up, proceed with
        # the analysis.

        # Create the branches in order
        branches = [corename, sunlocation, fits_level, wave, region]

        # Set up the roots we are interested in
        roots = {"pickle": ldirroot,
                 "image": sfigroot,
                 "csv": scsvroot}

        # Data and save locations are based here
        locations = datatools.save_location_calculator(roots,
                                     branches)

        # set the saving locations
        sfig = locations["image"]
        scsv = locations["csv"]

        # Identifier
        ident = datatools.ident_creator(branches)

        # Go through all the windows
        for iwindow, window in enumerate(windows):
            # Which window
            print('Window: ' + window + ' (%i out of %i)' % (iwindow + 1, len(windows)))

            # Update the region identifier
            region_id = '.'.join((ident, window, manip))

            # Create a location to save the figures
            savefig = os.path.join(os.path.expanduser(sfig), window, manip)
            if not(os.path.isdir(savefig)):
                os.makedirs(savefig)
            savefig = os.path.join(savefig, region_id)

            #for obstype in ['.iobs', '.logiobs']:
            for obstype in ['.logiobs']:
                #
                print('Analyzing ' + obstype)
                if obstype == '.iobs':
                    print('Data is lognormally distributed.')
                else:
                    print('Data is normally distributed.')

                # Load indices data
                pkl_location = locations['image'] + '/hanning/relative/'
                ifilename = region_id + '_M1_1_.logiobs__1d__.png.power_law_index'
                npy_file = os.path.join(pkl_location, ifilename + '.npy')
                print('Loading ' + npy_file)
                loaded_results = np.load(npy_file)

                # Get the data we are analyzing
                index_results[wave + region] = loaded_results

                # load the radial distances
                pkl_location = locations['pickle']
                ifilename = ident + '.mapcube.radial_distance'
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                this_radial_distance = pickle.load(pkl_file)
                pkl_file.close()
                # Store the radial distance
                radial_distance[wave].append(this_radial_distance)

                #M1_maximum_likelihood.' + region_id + '.pickle'
                pkl_location = locations['pickle']
                ifilename = 'M1_maximum_likelihood.' + region_id
                pkl_file_location = os.path.join(pkl_location, ifilename + '.pickle')
                print('Loading ' + pkl_file_location)
                pkl_file = open(pkl_file_location, 'rb')
                dummy = pickle.load(pkl_file)
                fitsummarydata = pickle.load(pkl_file)
                pkl_file.close()
                fsd[wave + region] = fitsummarydata


#
# Got all the data.  Now make the plots
#
fig, ax = plt.subplots(figsize=(10, 6))
ecolor = {'171': 'r', '193': 'b'}
fmt = {'171': 'ro', '193': 'bo'}
xytext = {'171': (0, 50), '193': (0, -50)}
for wave in waves:
    rchi2 = []

    for iregion, region in enumerate(regions):
        # Get the index value
        results = index_results[wave + region]
        indices[iregion] = results[0]
        # Get the 68% credible interval
        indices68[iregion, 0] = results[1]
        indices68[iregion, 1] = results[2]
        # Get the 95% credible interval
        indices95[iregion, 0] = results[3]
        indices95[iregion, 1] = results[4]
        # Get the reduced chi-squared
        rchi2.append(fsd[wave + region]['M1']['chi2'])

    yerr1 = indices[:] - indices68[:, 0]
    yerr2 = indices68[:, 1] - indices[:]
    r = np.asarray(radial_distance[wave]) / solar_radius_in_arcseconds
    #plt.plot(r, rchi2, color=ecolor[wave], linestyle=':', label=wave + ' reduced chi2')
    #plt.plot(indices68[:, 0], label=wave + ': 68%')
    #plt.plot(indices68[:, 1], label=wave + ': 68%')
    #plt.plot(indices95[:, 0], label=wave + ': 95%')
    #plt.plot(indices95[:, 1], label=wave + ': 95%')
    if sunlocation == 'equatorial':
        plt.axhline(ireland2014[wave]['qs'][0], label=wave + '$\AA$ quiet Sun (I2014)', color=ecolor[wave], linestyle='-.', linewidth=linewidth)
    if sunlocation == 'spoca665':
        plt.axhline(ireland2014[wave]['sunspot'][0], label=wave + '$\AA$ sunspot (I2014)', color=ecolor[wave], linestyle='-.', linewidth=linewidth)
        plt.axhline(ireland2014[wave]['loopfootpoints'][0], label=wave + '$\AA$ loop footpoints (I2014)', color=ecolor[wave], linestyle='--', linewidth=linewidth)
        plt.axhline(ireland2014[wave]['moss'][0], label=wave + '$\AA$ moss (I2014)', color=ecolor[wave], linestyle=':', linewidth=linewidth)
    plt.errorbar(r, indices, linestyle='-', yerr=[yerr1, yerr2], label=wave + '$\AA$ ' + sunday_name[sunlocation], fmt=fmt[wave], ecolor=ecolor[wave], capthick=2, linewidth=linewidth)

    plt.xlabel('distance (solar radii)')
    plt.ylabel('power law index')
    plt.title(sunday_name[sunlocation] + ': comparison of 171$\AA$ and 193$\AA$')


for iregion, region in enumerate(regions):
    plt.axvline(r[iregion], linestyle=":", color='y', linewidth=linewidth)
    plt.text(r[iregion], 1.55, region, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', fc='y', alpha=1.0))#, connectionstyle='arc3,rad=0'))


plt.legend(loc=0, framealpha=0.5)
plt.ylim(1.5, 4.0)
plt.tight_layout()
plt.show()
