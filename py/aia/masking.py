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
        within the bounds given by p.

        This uses a function which calculates the probability that a
        reduced chi-squared value exceeds a certain value.  Consider the default values
        p=(0.025, 0.975).  The value "0.975" means calculate the reduced chi-squared value for
        which there is a 97.5% chance that the actual value (rchi2) is above that level.  This must
        be a smaller value so that more of the probability distribution lies to the right of this
        value.

        Conversely, the value "0.025" means calculate the reduced chi-squared value for
        which there is a 2.5% chance that the actual value (rchi2) is above that level.  This must
        be a larger value so that more of the probability distribution lies to the left of this
        value.

        The actual measured value of rchi2 must lie between these limits in order for it to pass
        this test.  When this happens, the method returns True.  It returns False otherwise.

        Parameters
        ----------
        p : 2-element array

        Returns
        -------
        Returns True if the reduced chi-squared value lies within the specified probability limits.
        """
        if p[1] <= p[0]:
            raise ValueError(' The elements of "p" must be strictly monotonic.')
        if p[0] < 0.0:
            raise ValueError(' The first element of "p" must be greater than or equal to zero.')
        if p[1] > 1.0:
            raise ValueError(' The last element of "p" must be less than or equal to 1.')
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
    def __init__(self, data, q=None, absolute_level=None, area=((0, 10), (0, 10))):
        """
        Returns Boolean arrays
        :param data:
        :param bounds:
        """
        self.data = data
        self.q = q
        self.absolute_level = absolute_level
        self.area = area
        if self.q is None and self.absolute_level is None:
            raise ValueError('Either "q" or "absolute_level" must not None.')
        if self.q is not None and self.absolute_level is not None:
            raise ValueError('Only one of "q" or "absolute_level" can be set.')
        if self.q is not None:
            self._level = np.nanquantile(self.data, self.q)
        if self.absolute_level is not None:
            self._level = absolute_level

    @property
    def level(self):
        return self._level

    @property
    def mask(self):
        return self.data < self._level
