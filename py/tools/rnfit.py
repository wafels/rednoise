"""
Given a time series, fit the given PyMC model to it
"""
import numpy as np
import pymc
import pickle
import matplotlib.pyplot as plt


class Do_MCMC:
    def __init__(self, data, dt, bins=100, nsample=1):
        self.data = data
        self.dt = dt
        self.shape = data.shape
        self.nt = self.shape[-1]
        self.t = self.dt * np.arange(0, self.nt)
        # FFT frequencies
        self.f = np.fft.fftfreq(self.nt, self.dt)
        # positive frequencies
        self.fpos_index = self.f > 0
        self.fpos = self.f[self.fpos_index]
        self.bins = bins
        self.nsample = nsample

    # Do the PyMC fit
    def okgo(self, pymcmodel, locations=None, **kwargs):
        """Controls the PyMC fit of the input data"""

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
            ts = self._get_ts(k, loc)
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
        return self

    def save(self, filename='Do_MCMC_output.pickle'):
        """Save the results to a pickle file"""
        self.filename = filename
        print 'Saving to ' + self.filename
        output = open(self.filename, 'wb')
        pickle.dump(self.data, output)
        pickle.dump(self.locations, output)
        pickle.dump(self.results, output)
        output.close()

    def showfit(self, loc=0, figure=2):
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
                   label='norm. obs. power')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['mean'],
                   label='mean')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['quantiles'][50],
                   label='median')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['95% HPD interval'][:, 0],
                   label='low, 95% HPD')
        plt.loglog(x,
                   self.results[loc]['stats']['fourier_power_spectrum']['95% HPD interval'][:, 1],
                   label='high, 95% HPD')
        plt.xlabel('frequencies (Hz)')
        plt.ylabel('normalized power')
        plt.title('model fit')
        nd = len(description)
        ymax = np.log(np.max(self.results[loc]["power"]))
        ymin = np.log(np.min(self.results[loc]["power"]))
        ymax = ymin + 0.5 * (ymax - ymin)
        ystep = (ymax - ymin) / (1.0 * nd)
        for i, d in enumerate(description):
            ypos = np.exp(ymin + i * ystep)
            plt.text(x[0], ypos, d, fontsize=8)
        plt.legend(fontsize=10)
        plt.show()

    def showdeviation(self, loc=0, figure=2):
        """ Show the scaled deviation"""
        pwr = self.results[loc]["power"]
        mean = self.results[loc]['stats']['fourier_power_spectrum']['mean']
        deviation = (pwr - mean) / mean
        self._calculate_mss()
        plt.figure(figure)
        plt.semilogx(self.results[loc]["frequencies"], deviation, label='scaled deviation')
        plt.axhline(y=0, label='no deviation', color='k')
        plt.axhline(y=np.sqrt(self.mss[loc]), color='g', label='RMS scaled deviations = %8.4f' % (np.sqrt(self.mss[loc])))
        plt.axhline(y=-np.sqrt(self.mss[loc]), color='g')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('scaled deviation: (power - mean fit) / mean fit')
        plt.legend(fontsize=10)
        plt.show()

    def showts(self, loc=0, figure=1):
        """ Show the time-series """
        plt.figure(figure)
        plt.plot(self.t, self._get_ts(loc, loc), label='time series')
        plt.xlabel('time (seconds)')
        plt.ylabel('emission')
        plt.legend(fontsize=10)
        plt.show()

    def showall(self, loc=0):
        """Shows all the sumamry plots"""
        self.showts(loc=loc, figure=1)
        self.showfit(loc=loc, figure=2)
        self.showdeviation(loc=loc, figure=3)

    #Get a time series from the input data
    def _get_ts(self, k, loc):
        """ Get a time series from the input data"""
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

        return ts

    def _calculate_mss(self):
        """ Calculate the mean of the sum of square scaled deviations"""
        self.mss = []
        for r in self.results:
            pwr = r["power"]
            mean = r['stats']['fourier_power_spectrum']['mean']
            self.mss.append(np.sum(((pwr - mean) / mean) ** 2) / (1.0 * (np.size(pwr))))

