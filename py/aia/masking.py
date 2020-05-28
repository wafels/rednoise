"""
Calculates masks for the results arrays.
"""
import numpy as np
from tools import lnlike_model_fit



class Fitness:
    def __init__(self, observed_power, bestfit, k):
        self.observed_power = observed_power
        self.bestfit = bestfit
        self.n = len(observed_power)
        self.k = k
        self._dof = self.n - self.k - 1

        self._rhoj = lnlike_model_fit.rhoj(self.observed_power, self.bestfit)
        self._rchi2 = lnlike_model_fit.rchi2(1.0, self._dof, self._rhoj)

    def _rchi2limit(self, p):
        return lnlike_model_fit.rchi2_given_prob(p, 1.0, self._dof)

    def is_good(self, p=(0.025, 0.975)):
        """
        Tests the probability that the reduced-chi squared value is
        within the bounds given by p

        Parameters
        ----------
        p : 2-element array

        Returns
        -------
        Returns True if the reduced chi-squared value lies within
        the specified probability limits
        """
        rchi2_gt_low_limit = self._rchi2 > self._rchi2limit(p[1])
        rchi2_lt_high_limit = self._rchi2 < self._rchi2limit(p[0])
        return rchi2_gt_low_limit * rchi2_lt_high_limit


class VariableBounds(object):
    def __init__(self, data, bounds):
        """
        Returns Boolean arrays
        :param data:
        :param bounds:
        """
        self.data = data
        self.bounds = bounds

    @property
    def exceeds_low(self):
        if self.bounds[0] is None:
            return np.zeros_like(self.data, dtype=bool)
        else:
            return self.data < float(self.bounds[0])

    @property
    def exceeds_high(self):
        if self.bounds[1] is None:
            return np.zeros_like(self.data, dtype=bool)
        else:
            return self.data > float(self.bounds[1])


class IntensityMask(object):
    def __init__(self, data, p):
        """
        Returns Boolean arrays
        :param data:
        :param bounds:
        """
        self.data = data
        self.p = p
        self._nanquantile = np.nanquantile(self.data, self.p)

    @property
    def mask(self):
        return self.data < self._nanquantile
