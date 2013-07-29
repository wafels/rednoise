"""
Given a time series, fit the given PyMC model to it
"""
import numpy as np
import pymc
import pickle
import pymcmodels

class Do_MCMC:
    def __init__(self, data, dt, bins=100, nsample=1):
        self.data = data
        self.dt = dt
        self.shape = data.shape
        self.nt = self.shape[-1]
        # FFT frequencies
        self.f = np.fft.fftfreq(self.nt, self.dt)
        # positive frequencies
        self.fpos_index = self.f > 0
        self.fpos = self.f[self.fpos_index]
        self.bins = bins
        self.nsample = nsample

    def okgo(self, locations=None, bins=100,
             pli_range=[-1.0, 6.0],
             nor_range=[-10, 10],
             bac_range=[-20, 20], **kwargs):

        self.pli_range = pli_range
        self.nor_range = nor_range
        self.bac_range = bac_range

        if locations is None or len(self.data) == 1:
            self.locations = [1]
        else:
            self.locations = locations

        # Define the number of results we are looking at
        self.nsample = len(self.locations)
        self.results = np.zeros(shape=(3, self.nsample, self.bins))

        for k, loc in enumerate(self.locations):
            # Progress
            print(' ')
            print('Location number %i of %i' % (k + 1, self.nsample))

            # Get the time series
            if len(self.data.shape) == 1:
                ts = self.data

            if len(self.data.shape) == 2:
                y = loc[0]
                ts = self.data[y, :]
                print('Pixel Location (y) = %i' % (y))

            if len(self.data.shape) == 3:
                y = loc[0]
                x = loc[1]
                ts = self.data[y, x, :]
                print('Pixel Location x,y = %i, %i' % (x, y))

            # Do the MCMC
            # Calculate the power at the positive frequencies
            pwr = ((np.abs(np.fft.fft(ts))) ** 2)[self.fpos_index]
            # Normalize the power
            pwr = pwr / pwr[0]
            # Get the PyMC model
            pwr_law_with_constant = pymcmodels.single_power_law_with_constant(self.fpos, pwr)
            # Set up the MCMC model
            M = pymc.MCMC(pwr_law_with_constant)
            # Do the MAP calculation
            M.sample(**kwargs)

            q1 = M.trace("power_law_index")[:]
            self.results[0, k, :] = np.histogram(q1, bins=self.bins, range=self.pli_range)[0]

            q2 = M.trace("power_law_norm")[:]
            self.results[1, k, :] = np.histogram(q2, bins=self.bins, range=self.nor_range)[0]

            q3 = M.trace("background")[:]
            self.results[2, k, :] = np.histogram(q3, bins=self.bins, range=self.bac_range)[0]

        self.epli = np.histogram(q1, bins=self.bins, range=self.pli_range)[1]
        self.epln = np.histogram(q2, bins=self.bins, range=self.nor_range)[1]
        self.ebac = np.histogram(q3, bins=self.bins, range=self.bac_range)[1]

        return self

    def save(self, filename='Do_MCMC_output.pickle'):
        self.filename = filename
        print 'Saving to ' + self.filename
        output = open(self.filename, 'wb')
        pickle.dump(self.data, output)
        pickle.dump(self.locations, output)
        pickle.dump(self.results, output)
        pickle.dump(self.epli, output)
        pickle.dump(self.epln, output)
        pickle.dump(self.ebac, output)
        output.close()
