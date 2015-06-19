#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import astropy.units as u
import analysis_get_data
import study_details as sd
from analysis_details import rchi2limitcolor, limits, get_mask_info, get_image_model_location


class Explore:
    def __init__(self,
                 waves=('171', '193'),
                 regions=('sunspot', 'moss', 'quiet Sun', 'loop footpoints'),
                 windows=('hanning',)):

        # Waves
        self.waves = waves

        # Regions
        self.regions = regions

        # Apodization windows
        self.windows = windows

        # Get the data
        self.storage = analysis_get_data.get_all_data(waves=self.waves,
                                                      regions=self.regions)

        # Available models
        self._available_models = ???

        # Common parameters in all models
        self._common_parameters = ???

        # Good fit masks
        self.good_fit_masks = {}
        for wave in self.waves:
            self.good_fit_masks[wave] = {}
            for region in self.regions:
                self.good_fit_masks[wave][region] = {}
                for model in self._available_models:
                    self.good_fit_masks[wave][region][model] = self.storage[wave][region][model].good_fits()

        # Parameter limit masks
        self.parameter_limit_masks = {}
        for wave in self.waves:
            self.parameter_limit_masks[wave] = {}
            for region in self.regions:
                self.parameter_limit_masks[wave][region] = {}
                for model in self._available_models:
                    self.parameter_limit_masks[wave][region][model] = {}
                    this = self.storage[wave][region][model]
                    for parameter in this.parameters:
                        p = this.as_array(parameter)
                        p_mask = np.zeros_like(p, dtype=bool)
                        p_mask[np.where(p < limits[parameter][0])] = True
                        p_mask[np.where(p > limits[parameter][1])] = True
                        self.parameter_limit_masks[wave][region][model][parameter] = p_mask


        # IC data
        # Each model has an information quantity calculated.  The minimum value
        # indicates the preferred model.
        self.ic_data = {}
        for wave in self.waves:
            self.ic_data[wave] = {}
            for region in self.regions:
                self.ic_data[wave][region] = {}
                for model in self._available_models:
                    self.ic_data[wave][region][model] = {}
                    this = self.storage[wave][region][model]
                    for ic in ('AIC', 'BIC'):
                        self.ic_data[wave][region][model][ic] = this.as_array(ic)

        # Map information


        # Plotting information
        self._linewidth = 3

        # Oscillation locations
        self.five_minute = {"period": 300 * u.s,
                            "linewidth": self._linewidth,
                            "linecolor": 'k',
                            "linestyle": "-",
                            "label": "5 minutes"}

        self.three_minute = {"period": 180 * u.s,
                             "linewidth": self._linewidth,
                             "linecolor": 'k',
                             "linestyle": ":",
                             "label": "3 minutes"}

    def preferred_model_mask(self, preferred_model, ic_type, ic_limit):
        mask = {}
        for wave in self.waves:
            mask[wave] = {}
            for region in self.regions:
                mask[wave][region] = 0.0

                # Get the IC of the preferred model
                preferred_ic = self.ic_data[wave][region][preferred_model][ic_type]

                # Mask definition
                final_mask = np.ones_like(preferred_ic, dtype=bool)

                # Go through all the preferred models
                for model in self._available_models:
                    # Exclude the preferred model as it will by definition
                    # satisfy the limit criterion
                    if model != preferred_model:
                        # Difference between the preferred model IC and the
                        # current model IC
                        diff = preferred_ic - self.ic_data[wave][region][model][ic_type]
                        # The preferred mask assumes that none of the locations
                        # are preferred.
                        preferred_mask = np.zeros_like(preferred_ic, dtype=bool)
                        # Find the locations of the preferred model
                        preferred_mask[np.where(diff <= ic_limit)] = True
                        # The final mask - keep the pixels in the preferred
                        # model that really do have the lowest value compared
                        # to all the other mode ICs
                        final_mask = np.logical_and(final_mask, preferred_mask)
                # Store in a form usable for numpy masked arrays.
                mask[wave][region] = np.logical_not(final_mask)
        return mask

    def get_all_mask(self, wave, region, this_model, parameter_name, ic_type, ic_limit):
        mask_gfm1 = self.good_fit_masks()[wave][region][this_model]
        mask_ev = self.parameter_limit_masks()[wave][region][this_model][parameter_name]
        mask_ic = self.preferred_model_mask(this_model, ic_type, ic_limit)

        return np.logical_or(np.logical_or(mask_gfm1, mask_ev), mask_ic)

    def polygon_mask(self, polygon=None):
        """

        :param polygon: a spatial polygon defining an area of interest in the
        data.  If set to none, use the full spatial extent of the data
        :return:
        """
        pass

    # Plot cross-correlations across different AIA channels
    def two_d_cc_across_wave_spectral_model_parameter(self,
                                                      model_names=('Power law + Constant + Lognormal', 'Power law + Constant'),
                                                      ic_types={'none': None, 'AIC': 0.0, 'BIC': 6.0},
                                                      bins=100):


        plot_type = 'cc.across'

        # First wave
        for iwave1, wave1 in enumerate(self.waves):

            # Second wave
            for iwave2 in range(iwave1+1, len(self.waves)):
                wave2 = self.waves[iwave2]

                # Select a region
                for region in self.regions:

                    # branch location
                    b = [sd.corename, sd.sunlocation, sd.fits_level, wave1, region]

                    # Region identifier name
                    region_id = sd.datalocationtools.ident_creator(b)

                    # Output location
                    output = sd.datalocationtools.save_location_calculator(sd.roots, b)["pickle"]

                    # Different information criteria
                    for ic_type in ic_types:
                        for this_model in model_names:
                            this1 = self.storage[wave1][region][this_model]
                            this2 = self.storage[wave2][region][this_model]
                            parameters = this1.model.parameters
                            npar = len(parameters)
                            image = get_image_model_location(sd.roots, b, [this_model, ic_type])

                            # First parameter
                            for i in range(0, npar):
                                # First parameter name
                                p1_name = parameters[i]
                                # First parameter, label for the plot
                                p1_label = this1.model.labels[i]
                                # First parameter mask
                                p1_mask = self.get_all_mask(wave1, region, this_model, p1_name, ic_type, ic_limit)

                                # Second parameter
                                for j in range(0, npar):
                                    # Second parameter name
                                    p2_name = parameters[j]
                                    # Second parameter, label for the plot
                                    p2_label = this2.model.labels[j]
                                    # First parameter, data
                                    p2 = this2.as_array(p2_name)
                                    # Second parameter mask
                                    p2_mask = self.get_all_mask(wave2, region, this_model, p1_name, ic_type, ic_limit)

                                    # Final mask for cross-correlation
                                    final_mask = np.logical_not(np.logical_not(p1_mask) * np.logical_not(p2_mask))
                                    title = region + get_mask_info(final_mask) + '%s\n%s' % (ic_type, this_model)

                                    # Get the data using the final mask
                                    p1 = np.ma.array(this1.as_array(p1_name), mask=final_mask).compressed()
                                    p2 = np.ma.array(this2.as_array(p2_name), mask=final_mask).compressed()

                                    # Cross correlation statistics
                                    r = [spearmanr(p1, p2), pearsonr(p1, p2)]

                                    # Form the rank correlation string
                                    rstring = 'spr=%1.2f_pea=%1.2f' % (r[0][0], r[1][0])

                                    # Identifier of the plot
                                    plot_identity = rstring + '.' + region + '.' + wave1 + '.' + p1_name + '.' +  wave2 + '.' + p2_name + '.' + ic_type

                                    # Make a scatter plot
                                    xlabel = p1_label + r'$_{%s}$' % wave1
                                    ylabel = p2_label + r'$_{%s}$' % wave2
                                    title = '%s vs. %s, %s' % (xlabel, ylabel, title)
                                    plt.close('all')
                                    plt.title(title)
                                    plt.xlabel(xlabel)
                                    plt.ylabel(ylabel)
                                    plt.scatter(p1, p2)
                                    plt.xlim(limits[p1_name])
                                    plt.ylim(limits[p2_name])
                                    x0 = plt.xlim()[0]
                                    ylim = plt.ylim()
                                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                                    plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                                    plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                                    if p1_name == p2_name:
                                        plt.plot([limits[p1_name][0], limits[p1_name][1]],
                                                 [limits[p1_name][0], limits[p1_name][1]],
                                                 color='r', linewidth=3,
                                                 label='%s=%s' % (xlabel, ylabel))
                                    ofilename = this_model + '.' + plot_type + '.' + plot_identity + '.scatter.png'
                                    plt.legend(framealpha=0.5)
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(image, ofilename))

                                    # Make a 2d histogram
                                    plt.close('all')
                                    plt.title(title)
                                    plt.xlabel(xlabel)
                                    plt.ylabel(ylabel)
                                    plt.hist2d(p1, p2, bins=bins)
                                    x0 = plt.xlim()[0]
                                    ylim = plt.ylim()
                                    y0 = ylim[0] + 0.3 * (ylim[1] - ylim[0])
                                    y1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
                                    plt.text(x0, y0, 'Pearson=%f' % r[0][0], bbox=dict(facecolor=rchi2limitcolor[1], alpha=0.5))
                                    plt.text(x0, y1, 'Spearman=%f' % r[1][0], bbox=dict(facecolor=rchi2limitcolor[0], alpha=0.5))
                                    if p1_name == p2_name:
                                        plt.plot([limits[p1_name][0], limits[p1_name][1]],
                                                 [limits[p1_name][0], limits[p1_name][1]],
                                                 color='r', linewidth=3,
                                                 label="%s=%s" % (xlabel, ylabel))
                                    ofilename = this_model + '.' + plot_type + '.' + plot_identity + '.hist2d.png'
                                    plt.legend(framealpha=0.5)
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(image, ofilename))
