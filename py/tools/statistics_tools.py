#
# Summary statistics tools
#
import numpy as np


class Summary:
    def __init__(self, data, axis=None, q=[2.5, 97.5]):

        self.data = np.asarray(data)

        # Which axis to go over
        self.axis = axis

        # Percentiles
        self.q = q

        # Summary stats
        self.median = np.nanmedian(self.data, axis=self.axis)
        self.mean = np.nanmean(self.data, axis=self.axis)
        self.percentile = np.nanpercentile(self.data, self.q, axis=self.axis)
        self.min = np.nanmin(self.data, axis=self.axis)
        self.max = np.nanmax(self.data, axis=self.axis)
        self.var = np.nanvar(self.data, axis=self.axis)
        self.mad = np.nanmedian(np.abs(self.median - self.data), axis=self.axis)
        self.n_nans = np.sum(np.isnan(self.data))
