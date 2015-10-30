#
# Analysis - distributions.  Load in all the data and make some population
# and spatial distributions
#
import numpy as np


class MaskDefine:
    def __init__(self, storage, limits):
        """
        Defines masks for a given storage array.

        :param storage:
        :return:
        """
        #

        # Waves
        self.waves = storage.keys()
        w0 = self.waves[0]

        # Regions
        self.regions = storage[w0].keys()
        r0 = self.regions[0]

        # Available models.  For the IC limit stuff to work the code is much
        # easier if the results at the model level are stored as OrderedDict.
        self.available_models = storage[w0][r0].keys()

        # Common parameters in all models
        self.common_parameters = set(v.fit_parameter for v in storage[w0][r0][self.available_models[0]].model.variables)
        for model in self.available_models:
            par = [v.fit_parameter for v in storage[w0][r0][model].model.variables]
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
        #
        # parameter_limit_masks - masks that indicate where limits have been
        #                         exceeded, for every parameter
        # all_parameter_limit_masks - masks that indicate pixels where at least
        #                             one of the parameter limits have been
        #                             exceeded.
        # combined_good_fit_parameter_limit - masks that indicate where a bad
        #                                     fit has occurred, or where at
        #                                     least one of the parameter limits
        #                                     has been exceeded.  These masks
        #                                     give the values you want to work
        #                                     with in further analysis
        self.parameter_limit_masks = {}
        self.all_parameter_limit_masks = {}
        self.combined_good_fit_parameter_limit = {}
        for wave in self.waves:
            self.parameter_limit_masks[wave] = {}
            self.all_parameter_limit_masks[wave] = {}
            self.combined_good_fit_parameter_limit[wave] = {}
            for region in self.regions:
                self.parameter_limit_masks[wave][region] = {}
                self.all_parameter_limit_masks[wave][region] = {}
                self.combined_good_fit_parameter_limit[wave][region] = {}
                for model in self.available_models:

                    # Get the parameter data
                    this = storage[wave][region][model]

                    # Go through each parameter and find where the limits are
                    # exceeded.
                    parameters = [v.fit_parameter for v in this.model.variables]

                    # Stores a mask such that True means that parameter value
                    # either exceeded its low or high limit
                    self.parameter_limit_masks[wave][region][model] = {}

                    # Stores a mask where True means that at least one parameter
                    # limit was exceeded in the model.
                    dummy_p_mask = np.zeros_like(this.as_array(parameters[0]).value, dtype=bool)
                    self.all_parameter_limit_masks[wave][region][model] = dummy_p_mask

                    # Stores a mask where True means that either (a) at least
                    # one parameter limit was exceeded, or (b) a bad fit was
                    # detected during the fir process
                    self.combined_good_fit_parameter_limit[wave][region][model] = dummy_p_mask

                    for parameter in parameters:
                        # Next parameter
                        p = this.as_array(parameter).value

                        # Default - all parameters values are NOT masked
                        p_mask = np.zeros_like(p, dtype=bool)

                        # Mask out where the limits are exceeded for this
                        # parameter and store it
                        p_mask[np.where(p < limits[parameter][0].value)] = True
                        p_mask[np.where(p > limits[parameter][1].value)] = True
                        self.parameter_limit_masks[wave][region][model][parameter] = p_mask

                        # This mask determines which positions have all their
                        # model parameter values within their specified limits.
                        self.all_parameter_limit_masks[wave][region][model] = np.logical_or(p_mask, self.all_parameter_limit_masks[wave][region][model])

                    # Combined good fit masks and parameter limit masks
                    # This combination is used frequently, and so is worth
                    # calculating explicitly
                    self.combined_good_fit_parameter_limit[wave][region][model] = np.logical_or(self.all_parameter_limit_masks[wave][region][model], self.good_fit_masks[wave][region][model])

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
        information criterion.
        :param ic_type: ic_type: information criterion we will use
        :param ic_limit:  the difference in the IC that the best model must
        exceed compared to the other models.
        :return: for each wave/region pair, a numpy masked array where the
        data array indicates which model is preferred.  The model is indicated
        by the key index to self.available_models.  For the mask, True indicates
        that the preferred model has NOT exceeded the information criterion
        limit.
        """
        preferred_model_index = {}
        for wave in self.waves:
            preferred_model_index[wave] = {}
            for region in self.regions:
                preferred_model_index[wave][region] = 0.0
                _shape = self.ic_data[wave][region][self.available_models[0]][ic_type].shape

                # Storage for the IC value for all models as a function of space
                this_ic = np.zeros((len(self.available_models), _shape[0], _shape[1]), dtype=np.float64)

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
        Return an array that indicates if this model is preferred at each pixel.
        :param ic_type: information criterion we will use
        :param ic_limit: the difference in the IC that the best model must
        exceed compared to the other models.
        :param this_model: the spectral model we are interested in.
        :return: for each wave/region pair, a numpy mask where True indicates
         that the requested model is not preferred.
        """
        pmi = self.which_model_is_preferred(ic_type, ic_limit)
        itmp = {}
        for wave in self.waves:
            itmp[wave] = {}
            for region in self.regions:
                itmp[wave][region] = np.zeros_like(pmi[self.waves[0]][self.regions[0]], dtype=bool)

                # Get the mask that indicates that the limit has been exceeded
                pmi_mask = np.ma.getmask(pmi[wave][region])

                # Find the model index
                model_index = self.available_models.index(this_model)

                # False means that the model at this pixel is the same as the
                # requested model
                this_model_here_mask = np.ma.getdata(pmi[wave][region]) != model_index

                # Return the index as to which model is preferred.  Masked
                # values indicate that the model with the minimum IC value has
                # NOT exceeded the IC limit criterion.  When using this, you
                # have to use the self.available_models, as this maintains the
                # ordering of the models used here (since storage uses an
                # OrderedDict at the model level).
                itmp[wave][region] = np.logical_or(pmi_mask, this_model_here_mask)

        return itmp
