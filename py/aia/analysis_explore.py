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


class MaskDefine:
    def __init__(self, storage):
        """
        Defines masks for a given storage array.

        :param storage:
        :return:
        """

        # Waves
        self.waves = storage.keys()

        # Regions
        self.regions = storage[self.waves[0]].keys()

        # Available models
        self.available_models = storage[self.waves[0]][self.regions[0]].keys()

        # Common parameters in all models
        self.common_parameters = set(storage[self.waves[0]][self.regions[0]][self.available_models[0]].model.parameters)
        for model in self.available_models:
            par = storage[self.waves[0]][self.regions[0]][model].model.parameters
            self.common_parameters = self.common_parameters.intersection(par)

        # Good fit masks.
        # Find where the fitting algorithm has worked, and
        # a reasonable value of reduced chi-squared has been achieved.  Points
        # which have a bad fit are labeled True.
        self.good_fit_masks = {}
        for wave in self.waves:
            self.good_fit_masks[wave] = {}
            for region in self.regions:
                self.good_fit_masks[wave][region] = {}
                for model in self.available_models:
                    self.good_fit_masks[wave][region][model] = storage[wave][region][model].good_fits()

        # Parameter limit masks.
        # Find where the parameters exceed limits, and create a mask that can
        # be used with numpy masked arrays.  Points in the mask marked True
        # exceed the parameter limits.
        self.parameter_limit_masks = {}
        for wave in self.waves:
            self.parameter_limit_masks[wave] = {}
            for region in self.regions:
                self.parameter_limit_masks[wave][region] = {}
                for model in self.available_models:
                    self.parameter_limit_masks[wave][region][model] = {}
                    this = storage[wave][region][model]
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
                for model in self.available_models:
                    self.ic_data[wave][region][model] = {}
                    this = storage[wave][region][model]
                    for ic in ('AIC', 'BIC'):
                        self.ic_data[wave][region][model][ic] = this.as_array(ic)

    def which_model_is_preferred(self, ic_type, ic_limit):
        """
        Return an array that indicates which model is preferred according to the
        information criterion.
        :param ic_type:
        :param ic_limit:
        :return:
        """
        preferred_model_index = {}
        for wave in self.waves:
            preferred_model_index[wave] = {}
            for region in self.regions:
                preferred_model_index[wave][region] = np.zeros()

                # Storage for the IC value for all models as a function of space
                this_ic = np.zeros(len(self.available_models), self.nx, self.ny)

                # Fill up the array
                for imodel, model in enumerate(self.available_models):
                    this_ic[imodel, :, :] = self.ic_data[wave][region][model][ic_type]

                # The minimum value of the information criterion.
                pmi = np.argmin(this_ic, axis=0)

                # Sort so the first entry is the minimum value
                pmi_sort = np.sort(this_ic, axis=0)

                # Find where the minimum value is less than that required by
                # the IC limit
                test = pmi_sort[0, :, :] - pmi_sort[1, :, :] < -ic_limit

                # Return the index as to which model is preferred
                return np.ma.array(pmi, mask=np.logical_not(test))




        return preferred_model_index

    def preferred_model_mask(self, preferred_model, ic_type, ic_limit):
        """
        Return a mask that shows where a given is model according the specified information
        criterion.

        :param preferred_model: The spectral model that we are interested in.
        :param ic_type: The information criterion (IC) we wish to use
        :param ic_limit: The limit on the IC above which we expect the model to be preferred.
        :return: A numpy mask for every wave and region
        """
        mask = {}
        for wave in self.waves:
            mask[wave] = {}
            for region in self.regions:
                mask[wave][region] = 0.0

                # Get the IC of the preferred model
                preferred_ic = self.ic_data[wave][region][preferred_model][ic_type]

                # Mask definition
                final_mask = np.ones_like(preferred_ic, dtype=bool)

                # Go through all the models
                for model in self.available_models:
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
