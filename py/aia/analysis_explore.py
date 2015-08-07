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

        # Available models.  For the IC limit stuff to work the code is much
        # easier if the results at the model level are stored as OrderedDict.
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
        self.all_parameter_limit_masks = {}
        for wave in self.waves:
            self.parameter_limit_masks[wave] = {}
            self.all_parameter_limit_masks[wave] = {}
            for region in self.regions:
                self.parameter_limit_masks[wave][region] = {}
                self.all_parameter_limit_masks[wave][region] = {}
                for model in self.available_models:
                    self.parameter_limit_masks[wave][region][model] = {}
                    self.all_parameter_limit_masks[wave][region][model] = {}
                    this = storage[wave][region][model]
                    for parameter in this.parameters:
                        p = this.as_array(parameter)
                        p_mask = np.zeros_like(p, dtype=bool)
                        p_mask[np.where(p < limits[parameter][0])] = True
                        p_mask[np.where(p > limits[parameter][1])] = True
                        self.parameter_limit_masks[wave][region][model][parameter] = p_mask

                        # This mask determines which positions have all their
                        # parameter values within their specified limits.
                        self.all_parameter_limit_masks[wave][region][model] = np.logical_or(p_mask, self.all_parameter_limit_masks[wave][region][model])

        # IC data.
        # Extract the IC from the storage array.
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
        information criterion, at each pixel.
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

                # Find where difference between the minimum value and the next
                # largest is greater than that specified.
                test = pmi_sort[0, :, :] - pmi_sort[1, :, :] < -ic_limit

                # Return the index as to which model is preferred.  Masked
                # values indicate that the model with the minimum IC value has
                # NOT exceeded the IC limit criterion.  When using this, you
                # have to use the self.available_models, as this maintains the
                # ordering of the models used here (since storage uses an
                # OrderedDict at the model level).
                preferred_model_index[wave][region] = np.ma.array(pmi, mask=np.logical_not(test))

        return preferred_model_index

    def is_this_model_preferred(self, ic_type, ic_limit, this_model):
        """
        Return an array that indicates if this model is preferred at each pixel
        :param ic_type:
        :param ic_limit:
        :return:
        """
        pmi = self.which_model_is_preferred_at_each_pixel(ic_type, ic_limit)
        itmp = {}
        for wave in self.waves:
            itmp[wave] = {}
            for region in self.regions:
                itmp[wave][region] = np.zeros_like(pmi[self.waves[0]][self.regions[0]], dtype=bool)

                # Get the mask that indicates that the limit has been exceeded
                pmi_mask = np.ma.getmask(pmi[wave][region])

                # Find the model index
                model_index = self.available_models.index(this_model)

                this_model_here_mask = np.ma.getarray(pmi[wave][region]) != model_index

                # Return the index as to which model is preferred.  Masked
                # values indicate that the model with the minimum IC value has
                # NOT exceeded the IC limit criterion.  When using this, you
                # have to use the self.available_models, as this maintains the
                # ordering of the models used here (since storage uses an
                # OrderedDict at the model level).
                itmp[wave][region] = np.logical_or(pmi_mask, this_model_here_mask)

        return itmp
