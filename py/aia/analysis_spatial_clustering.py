#
# Analysis - spatial clustering
#
# Look for different clusters in the spectral information
#
import os
import glob
from copy import deepcopy
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import analysis_get_data
import details_study as ds
import details_analysis as da
import details_plots as dp


# Wavelengths we want to cross correlate
waves = ['131', '171', '193', '211', '335', '94']
waves = ['131', '171', '193', '211', '335']
#waves = ['171', '193']

# Regions we are interested in
# regions = ['sunspot', 'moss', 'quiet Sun', 'loop footpoints']
# regions = ['most_of_fov']
regions = ['six_euv']

use_position_offset = 0


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

# Get the common parameters
parameters = ['ln(power law amplitude)', 'power law index', 'ln(constant)']
parameters = ['power law index']
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
                    if len(directory_listing) >= 2:
                        raise ValueError("More than 1 file selected.")
                    elif len(directory_listing) < 1:
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

                        all_data = np.zeros(shape=(ny*nx, use_position_offset + len(waves)))
                        all_mask = np.zeros(shape=(ny*nx), dtype=bool)

                        # Create the positional information
                        all_data_x = np.zeros(shape=(ny, nx))
                        for j in range(0, ny):
                            all_data_x[j, :] = np.arange(0, nx)

                        all_data_y = np.zeros_like(all_data_x)
                        for i in range(0, nx):
                            all_data_y[:, i] = np.arange(0, ny)

                    # Update the mask
                    all_mask = np.logical_or(all_mask, submap.mask.flatten())

                    # Update the data array
                    all_data[:, iwave] = submap.data.flatten()

                #
                if use_position_offset == 2:
                    all_data[:, len(waves)] = all_data_x.flatten()
                    all_data[:, len(waves) + 1] = all_data_y.flatten()

                # Number of features
                n_features = all_data.shape[1]

                # Number of
                nleft = np.sum(np.logical_not(all_mask))
                print("Number of data points is %i (%f%%)" % (nleft, 100*nleft/(1.0*nx*ny)))
                all_masked_data = np.zeros((nleft, use_position_offset + len(waves)))
                for n in range(0, n_features):
                    all_masked_data[:, n] = np.ma.array(all_data[:, n], mask=all_mask).compressed()

                # Run the clustering algorithm
                kmeans = KMeans()
                kmeans.fit(all_masked_data)

                #all_data2 = np.zeros((nx*ny, len(waves)))
                #for iwave in range(0, len(waves)):
                #    all_data2[:, iwave] = all_data[:, iwave]
                #    all_data2[np.where(all_mask), iwave] = -1


                # Run the clustering algorithm
                #kmeans2 = KMeans(n_clusters=5)
                #kmeans2.fit(all_data2)
                #kmeans_keep[ic_limit][parameter] = (kmeans2, all_data2)

                use_this = StandardScaler().fit_transform(deepcopy(all_masked_data))

                X_reduced = PCA(n_components=1).fit(use_this)

                range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
                for n_clusters in range_n_clusters:
                    print(n_clusters)
                    # Create a subplot with 1 row and 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.set_size_inches(18, 7)

                    # The 1st subplot is the silhouette plot
                    # The silhouette coefficient can range from -1, 1 but in this example all
                    # lie within [-0.1, 1]
                    ax1.set_xlim([-1, 1])
                    # The (n_clusters+1)*10 is for inserting blank space between silhouette
                    # plots of individual clusters, to demarcate them clearly.
                    ax1.set_ylim([0, 1000 + (n_clusters + 1) * 10])

                    # Initialize the clusterer with n_clusters value and a random generator
                    # seed of 10 for reproducibility.
                    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                    cluster_labels = clusterer.fit_predict(use_this)

                    # The silhouette_score gives the average value for all the samples.
                    # This gives a perspective into the density and separation of the formed
                    # clusters
                    n_select = 1
                    silhouette_avg_keep = []
                    for nrandom in range(0, n_select):
                        random_selection = np.random.random_integers(0, use_this.shape[0]-1, size=1000)
                        X = use_this[random_selection, :]
                        CL = cluster_labels[random_selection]
                        silhouette_avg_keep.append(silhouette_score(X, CL))
                    silhouette_avg = np.mean(np.asarray(silhouette_avg_keep))
                    print("For n_clusters =", n_clusters,
                          "The average silhouette_score is :", silhouette_avg)

                    # Compute the silhouette scores for each sample
                    sample_silhouette_values = silhouette_samples(X, CL)

                    y_lower = 10
                    for i in range(n_clusters):
                        # Aggregate the silhouette scores for samples belonging to
                        # cluster i, and sort them
                        ith_cluster_silhouette_values = sample_silhouette_values[CL == i]

                        ith_cluster_silhouette_values.sort()

                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i

                        color = cm.spectral(float(i) / n_clusters)
                        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                          0, ith_cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)

                        # Label the silhouette plots with their cluster numbers at the middle
                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                        # Compute the new y_lower for next plot
                        y_lower = y_upper + 10  # 10 for the 0 samples

                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")

                    # The vertical line for average silhoutte score of all the values
                    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                    #ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    # 2nd Plot showing the actual clusters formed
                    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
                    ax2.scatter(use_this[:, 0], use_this[:, 1], marker='.',
                                s=30, lw=0, alpha=0.7, c=colors)

                    # Labeling the clusters
                    centers = clusterer.cluster_centers_
                    # Draw white circles at cluster centers
                    ax2.scatter(centers[:, 0], centers[:, 1],
                                marker='o', c="white", alpha=1, s=200)

                    for i, c in enumerate(centers):
                        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

                    ax2.set_title("The visualization of the clustered data.")
                    ax2.set_xlabel("Feature space for the 1st feature")
                    ax2.set_ylabel("Feature space for the 2nd feature")

                    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                                  "with n_clusters = %d" % n_clusters),
                                 fontsize=14, fontweight='bold')

                    plt.show()