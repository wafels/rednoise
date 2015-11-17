#
# Analysis - spatial clustering
#
# Look for different clusters in the spectral information
#
import os
import glob
import cPickle as pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import analysis_get_data
import analysis_explore
import details_study as ds
import details_analysis as da
import details_plots as dp

import numpy as np
from sklearn.cluster import KMeans

# Wavelengths we want to cross correlate
waves = ['131', '171', '193', '211', '335', '94']
waves = ['131', '171', '193', '211', '335']

# Regions we are interested in
# regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
# regions = ['most_of_fov']
regions = ['six_euv']


# Apodization windows
windows = ['hanning']

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

#
# Details of the analysis
#
limits = da.limits['standard']
ic_types = da.ic_details.keys()

#
# Details of the plotting
#
fz = dp.fz
three_minutes = dp.three_minutes
five_minutes = dp.five_minutes
hloc = dp.hloc
linewidth = 3
bins = 100


# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names)
# Define the masks
mdefine = analysis_explore.MaskDefine(storage, limits)

# Get the common parameters
parameters = mdefine.common_parameters
npar = len(parameters)

# Get the sunspot outline
sunspot_outline = analysis_get_data.sunspot_outline()

# Plot spatial distributions of the common parameters
plot_type = 'spatial.clustering'

# Plot spatial distributions of the spectral model parameters.
# Different information criteria
for ic_type in ic_types:
    kmeans_keep = {}

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s>%f' % (ic_type, ic_limit)

        kmeans_keep[ic_limit] = {}
        # Select a region
        for region in regions:

            # Go through the common parameters and make a map.
            for parameter in parameters:

                # First wave
                for iwave, wave in enumerate(waves):

                    # branch location
                    b = [ds.corename, ds.sunlocation, ds.fits_level, wave, region]

                    # Region identifier name
                    region_id = ds.datalocationtools.ident_creator(b)

                    # Output location
                    output = ds.datalocationtools.save_location_calculator(ds.roots, b)["pickle"]
                    image = ds.datalocationtools.save_location_calculator(ds.roots, b)["image"]

                    # Output filename
                    ofilename = os.path.join(output, region_id + '.datacube')
                    glob_this = os.path.join(output, '*%s*%s*' % (parameter, ic_limit_string))

                    directory_listing = glob.glob(glob_this)
                    if len(directory_listing) >=2:
                        raise ValueError("More than 1 file selected.")
                    elif len(directory_listing) <1:
                        raise ValueError("No file selected.")

                    print("Loading file %s" % directory_listing[0])
                    f = open(directory_listing[0], 'rb')
                    submap = pickle.load(f)
                    f.close()

                    if iwave == 0:
                        ny = submap.data.shape[0]
                        y = np.arange(0, ny)

                        nx = submap.data.shape[1]
                        x = np.arange(0, nx)

                        all_data = np.zeros(shape=(ny*nx, len(waves)))
                        all_mask = np.zeros(shape=(ny*nx), dtype=bool)
                        #all_data[0, :] = np.repeat(x, ny)
                        #all_data[1, :] = np.tile(y, nx)

                    all_mask = np.logical_or(all_mask, submap.mask.flatten())
                    all_data[:, iwave] = submap.data.flatten()

                nleft = np.sum(np.logical_not(all_mask))
                print("Number of data points is %i (%f%%)" % (nleft, 100*nleft/(1.0*nx*ny)))
                all_masked_data = np.zeros((nleft, len(waves)))
                for iwave in range(0, len(waves)):
                    all_masked_data[:, iwave] = np.ma.array(all_data[:, iwave], mask=all_mask).compressed()

                # Run the clustering algorithm
                kmeans = KMeans()
                kmeans.fit(all_masked_data)

                all_data2 = np.zeros((nx*ny, len(waves)))
                for iwave in range(0, len(waves)):
                    all_data2[:, iwave] = all_data[:, iwave]
                    all_data2[np.where(all_mask), iwave] = -1


                # Run the clustering algorithm
                kmeans2 = KMeans(n_clusters=5)
                kmeans2.fit(all_data2)
                kmeans_keep[ic_limit][parameter] = (kmeans2, all_data2)