#
# Analysis - Plot the spatial distributions of spectral model parameters
#
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.units as u
from sunpy.time import parse_time
from sunpy.map import Map

import analysis_get_data
import analysis_explore
import details_study as ds
import details_analysis as da
import details_plots as dp
from tools import statistics_tools

# Wavelengths we want to cross correlate
waves = ['94', '131', '171', '193', '211', '335']

regions = ['six_euv']
#waves = ['193']
#regions = ['ch']

power_type = 'fourier_power_relative'
limit_type = 'standard'

# Apodization windows
windows = ['hanning']

fake_map = True

# Model results to examine
model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')
#model_names = ('Power Law + Constant + Lognormal', 'Power Law + Constant')

#
# Details of the analysis
#
limits = da.limits[limit_type]
ic_types = da.ic_details.keys()

#
# Details of the plotting
#
fz = dp.fz
three_minutes = dp.three_minutes
five_minutes = dp.five_minutes
bins = dp.histogram_1d_bins
fontsize = dp.fontsize

# Load in all the data
storage = analysis_get_data.get_all_data(waves=waves,
                                         regions=regions,
                                         model_names=model_names,
                                         spectral_model='.rnspectralmodels4')
mdefine = analysis_explore.MaskDefine(storage, limits)

# Plot cross-correlations across different AIA channels
plot_type = 'spatial.within'

# Plot spatial distributions of the spectral model parameters.
# Different information criteria
for ic_type in ic_types:

    # Get the IC limit
    ic_limits = da.ic_details[ic_type]
    for ic_limit in ic_limits:
        ic_limit_string = '%s>%f' % (ic_type, ic_limit)

        # Model name
        for this_model in model_names:

            # Select a region
            for region in regions:

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

                    # Get the data for this model
                    this = storage[wave][region][this_model]

                    # Parameters
                    parameters = [v.fit_parameter for v in this.model.variables]
                    npar = len(parameters)

                    # Get the combined mask
                    mask1 = mdefine.combined_good_fit_parameter_limit[wave][region][this_model]

                    # Get the region submap
                    if not fake_map:
                        if iwave == 0:
                            submaps = analysis_get_data.get_region_submap(output, region_id)
                            submap = submaps['reference region']
                    else:
                        fake_map_data = np.zeros_like(mask1, dtype=np.float)
                        fake_map_header = {"CDELT1": 0.6, "CDELT2": 0.6}
                        submap = Map((fake_map_data, fake_map_header))

                    for i in range(0, npar):
                        # Data
                        parameter = parameters[i]
                        p1 = this.as_array(parameter)

                        # Second parameter limits
                        p1_limits = limits[parameter]

                        # Label
                        label = this.model.variables[i].converted_label  # + r'$_{%s}$' % wave

                        # Find out if this model is preferred
                        mask2 = mdefine.is_this_model_preferred(ic_type, ic_limit, this_model)[wave][region]

                        # Final mask combines where the parameters are all nice,
                        # where a good fit was achieved, and where the IC limit
                        # criterion was satisfied.
                        final_mask = np.logical_or(mask1, mask2)

                        image = dp.get_image_model_location(ds.roots, b, [this_model, ic_limit_string, limit_type])

                        # Create the subtitle - model, region, information
                        # on how much of the field of view is not masked,
                        # and the information criterion and limit used.
                        number_pixel_string, percent_used_string, mask_info_string = dp.get_mask_info_string(final_mask)
                        ic_info_string = '%s, %s' % (ic_limit_string, mask_info_string)
                        subtitle = dp.concat_string([this_model,
                                                     '%s - %s' % (region, wave),
                                                     ic_info_string,
                                                     limit_type], sep='\n')
                        subtitle_filename = dp.concat_string([this_model,
                                                              region,
                                                              ic_limit_string,
                                                              limit_type], sep='.')

                        # Plot identity
                        plot_identity_filename = dp.concat_string([wave, parameter])

                        # Create the masked numpy array
                        map_data = ma.array(p1, mask=final_mask)

                        # Make a SunPy map for nice spatially aware plotting.
                        my_map = analysis_get_data.make_map(submap, map_data)

                        # Get the sunspot

                        # Get the feature/event data
                        """
                        fevent = (ds.fevents)[0]
                        fevent_filename = region_id + '.{:s}.{:s}.fevent{:s}.pkl'.format(fevent[0], fevent[1], ds.processing_info)
                        fevents = analysis_get_data.fevent_outline(None, None,
                                                                   None,
                                                                   fevent=None,
                                                                   download=False,
                                                                   directory=output,
                                                                   filename=fevent_filename)
                        """
                        # Make a spatial distribution map spectral model
                        # parameter
                        plt.close('all')
                        # Normalize the color table
                        norm = colors.Normalize(clip=False,
                                                vmin=p1_limits[0].value,
                                                vmax=p1_limits[1].value)

                        # Set up the palette we will use
                        palette = dp.spectral_parameters[parameter].cm

                        # Bad values are those that are masked out
                        palette.set_bad(dp.spectral_parameters[parameter].bad, 1.0)

                        # Begin the plot
                        fig, ax = plt.subplots()
                        # Plot the map
                        ret = my_map.plot(cmap=palette, axes=ax, interpolation='none',
                                          norm=norm)
                        title = '{:s}\n{:s}'.format(label, 'AIA '+wave+' $\AA$, ' + percent_used_string + ' fit')
                        ret.axes.set_title(title, fontsize=0.8*fontsize)

                        # Plot the features and events
                        dt = (30 * u.day).to(u.s).value
                        """
                        for i, fevent in enumerate(fevents):
                            this_dt = np.abs((parse_time(fevent.time) - submap.date).total_seconds())
                            if this_dt < dt:
                                this_fevent = i
                        z = fevents[this_fevent].solar_rotate(submap.date).mpl_polygon
                        z.set_edgecolor('r')
                        z.set_linewidth(1)
                        ax.add_artist(z)
                        """
                        cbar = fig.colorbar(ret, extend='both', orientation='vertical',
                                            shrink=0.8, label=label)
                        # Fit everything in.
                        ax.autoscale_view()

                        # Dump to file
                        final_filename = dp.concat_string([plot_type,
                                                           plot_identity_filename,
                                                           subtitle_filename]).replace(' ', '') + '.png'
                        final_filename = dp.clean_for_overleaf(final_filename)
                        filepath = os.path.join(image, final_filename)
                        print('Saving to ' + filepath)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close(fig)
