"""
Given a time series, fit the given PyMC model to it
"""
import numpy as np
import pymc
import pickle
import pymcmodels
import matplotlib.pyplot as plt

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

    def okgo(self, pymcmodel, locations=None, **kwargs):

        # stats results
        self.results = []

        # locations in the input array
        if locations is None or len(self.data) == 1:
            self.locations = [1]
        else:
            self.locations = locations

        # Define the number of results we are looking at
        self.nsample = len(self.locations)
        for k, loc in enumerate(self.locations):
            # Progress
            print(' ')
            print('Location number %i of %i' % (k + 1, self.nsample))

            # Get the time series
            if len(self.data.shape) == 1:
                ts = self.data

            if len(self.data.shape) == 2:
                ts = self.data[k, :]
                print('Entry number %i of %i' % (k + 1, self.nsample))

            if len(self.data.shape) == 3:
                y = loc[0]
                x = loc[1]
                ts = self.data[y, x, :]
                print('Pixel Location x,y = %i, %i' % (x, y))

            # Do the MCMC
            # Calculate the power at the positive frequencies
            self.pwr = ((np.abs(np.fft.fft(ts))) ** 2)[self.fpos_index]
            # Normalize the power
            self.pwr = self.pwr / self.pwr[0]
            # Set up the MCMC model
            self.pymcmodel = pymcmodel(self.fpos, self.pwr)
            self.M = pymc.MCMC(self.pymcmodel)
            # Do the MAP calculation
            self.M.sample(**kwargs)
            # Append the stats results
            self.results.append({"timeseries": ts,
                               "power": self.pwr,
                               "frequencies": self.fpos,
                               "location": loc,
                               "stats": self.M.stats()})
            """
            # MAP values
            M2 = pymc.MAP(self.pymcmodel)
            self.map_fit[0, k] = M2.power_law_index.value
            self.map_fit[1, k] = M2.power_law_norm.value
            self.map_fit[2, k] = M2.background.value

            q1 = M.trace("power_law_index")[:]
            self.results[0, k, :] = np.histogram(q1, bins=self.bins, range=self.pli_range)[0]

            q2 = M.trace("power_law_norm")[:]
            self.results[1, k, :] = np.histogram(q2, bins=self.bins, range=self.nor_range)[0]

            q3 = M.trace("background")[:]
            self.results[2, k, :] = np.histogram(q3, bins=self.bins, range=self.bac_range)[0]

        self.epli = np.histogram(q1, bins=self.bins, range=self.pli_range)[1]
        self.epln = np.histogram(q2, bins=self.bins, range=self.nor_range)[1]
        self.ebac = np.histogram(q3, bins=self.bins, range=self.bac_range)[1]
        """
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
        pickle.dump(self.map_fit, output)
        output.close()

    def showfit(self, loc=0, figure=1):
        """ Show a spectral fit summary plot"""
        # Construct a summary for each variab;e
        k = self.results[0]['stats'].keys()
        description = []
        for key in k:
            if key is not 'fourier_power_spectrum':
                mean = self.results[loc]['stats'][key]['mean']
                median = self.results[loc]['stats'][key]['quantiles'][50]
                hpd95 = self.results[loc]['stats'][key]['95% HPD interval']
                description.append(key + ': mean=%8.4f, median=%8.4f, 95%% HPD= %8.4f, %8.4f' % (mean, median, hpd95[0], hpd95[1]))
        
        x = self.results[loc]["frequencies"]
        
        plt.figure(figure)
        plt.loglog(x,
                   self.results[loc]["power"],
                   label = 'norm. obs.power')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['mean'],
                   label = 'mean')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['quantiles'][50],
                   label = 'median')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['95% HPD interval'][:, 0],
                   label = 'low, 95% HPD')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['95% HPD interval'][:, 1],
                   label = 'high, 95% HPD')
        
        
        plt.xlabel('frequencies (Hz)')
        plt.ylabel('normalized power')
        plt.title('model fit')
        nd = len(description)
        ymax = np.log(np.max(self.results[loc]["power"]))
        ymin = np.log(np.min(self.results[loc]["power"]))
        ymax = ymin + 0.5 * (ymax - ymin)
        ystep = (ymax - ymin)/(1.0 * nd)
        for i, d in enumerate(description):
            ypos = np.exp(ymin + i * ystep)
            plt.text(x[0], ypos, d, fontsize=8)
        plt.legend()
        plt.show()
        
        def showts(self, loc=0, figure=2):
            """ Show the original time series"""
            pass
        
        def showall(self, loc=0):
            self.showfit(loc=loc)
            self.showts(loc=loc)