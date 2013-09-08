"""
Given a time series, fit the given PyMC model to it
"""
import numpy as np
import pymc
import pickle
import matplotlib.pyplot as plt


class Do_MCMC:
    def __init__(self, data):
        """"
        Handles the MCMC fit of data.

        Parameters
        ----------
        data : a list of 1-d ndarrays, each of which is a time series to be
               analyzed
        dt : the cadence of the time-seris
        """
        self.data = data

        # Number of time series
        self.ndata = len(self.data)

    # Do the PyMC fit
    def okgo(self, pymcmodel, locations=None, **kwargs):
        """Controls the PyMC fit of the input data

        Parameters
        ----------
        pymcmodel : the PyMC model we are using
        locations : which elements of the input data we are analyzing
        **kwargs : PyMC control keywords
        """

        # stats results
        self.results = []

        # locations in the input array
        if locations is None:
            self.locations = range(0, self.ndata)
        else:
            self.locations = locations
        # Define the number of results we are looking at
        self.nts = len(self.locations)

        for k in self.locations:
            # Progress
            print(' ')
            print('Location number %i of %i' % (k + 1, self.nts))
            ts = self.data[k]
            # Do the MCMC
            # Calculate the power at the positive frequencies
            self.pwr = ts.PowerSpectrum.ppower
            self.fpos = ts.PowerSpectrum.frequencies.positive

            # Following Vaughan (2010): normalize to the mean power
            power_mean = np.mean(self.pwr)
            self.pwr = self.pwr / power_mean
            # Normalize now to the standard deviation
            power_std = np.std(self.pwr)
            self.pwr = self.pwr / power_std

            # Set up the MCMC model
            self.pymcmodel = pymcmodel(self.fpos, self.pwr)
            # Get the MAP results
            mp = pymc.MAP(self.pymcmodel)
            mp.fit(method='fmin_powell')

            self.M = pymc.MCMC(mp.variables)

            # Do the MCMC calculation
            self.M.sample(**kwargs)

            # Get the samples
            k = self.M.stats().keys()
            samples = {}
            for key in k:
                if key not in ('fourier_power_spectrum', 'predictive'):
                    samples[key] = self.M.trace(key)[:]

            # Append the stats results and the samples
            self.results.append({"timeseries": ts,
                                 "mean": power_mean,
                                 "std": power_std,
                                 "power": self.pwr,
                                 "frequencies": self.fpos,
                                 "location": k,
                                 "stats": self.M.stats(),
                                 "samples": samples,
                                 "mp": mp,
                                 "M": self.M})
        return self

    def save(self, filename='Do_MCMC_output.pickle'):
        """Save the results to a pickle file"""
        self.filename = filename
        print 'Saving to ' + self.filename
        self._calculate_mss()
        output = open(self.filename, 'wb')
        pickle.dump(self.data, output)
        pickle.dump(self.locations, output)
        pickle.dump(self.results, output)
        pickle.dump(self.mss, output)
        output.close()
        return self

    def showfit(self, loc=0, figure=2, show_cor_seis=True):
        """ Show a spectral fit summary plot"""
        # Construct a summary for each variable
        k = self.results[loc]['stats'].keys()
        description = []
        for key in k:
            if key not in ('fourier_power_spectrum', 'predictive'):
                stats = self.results[loc]['stats']
                mean = stats[key]['mean']
                median = stats[key]['quantiles'][50]
                hpd95 = stats[key]['95% HPD interval']
                description.append(key + ': mean=%8.4f, median=%8.4f, 95%% HPD= %8.4f, %8.4f' % (mean, median, hpd95[0], hpd95[1]))

        x = self.results[loc]["frequencies"]
        y = self.results[loc]["stats"]

        plt.figure(figure)
        plt.loglog(x,
                   self.results[loc]["power"],
                   label='norm. obs. power')
        plt.loglog(x, y['fourier_power_spectrum']['mean'], label='mean')
        plt.loglog(x, y['fourier_power_spectrum']['quantiles'][50],
                   label='median')
        plt.loglog(x, y['fourier_power_spectrum']['95% HPD interval'][:, 0],
                   label='low, 95% HPD')
        plt.loglog(x, y['fourier_power_spectrum']['95% HPD interval'][:, 1],
                   label='high, 95% HPD')
        if show_cor_seis:
            plt.axvline(1.0 / 300.0, color='k', linestyle='--', label='5 mins')
            plt.axvline(1.0 / 180.0, color='k', linestyle=':', label='3 mins')
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
        ts = self.data[loc]
        plt.figure(figure)
        plt.plot(ts.SampleTimes.time, ts.data, label='time series')
        plt.xlabel('time (seconds)')
        plt.ylabel('emission')
        plt.legend(fontsize=10)
        plt.show()

    def showall(self, loc=0):
        """Shows all the summary plots"""
        self.showts(loc=loc, figure=1)
        self.showfit(loc=loc, figure=2)
        self.showdeviation(loc=loc, figure=3)

    def _calculate_mss(self):
        """ Calculate the mean of the sum of square scaled deviations"""
        self.mss = []
        for r in self.results:
            pwr = r["power"]
            mean = r['stats']['fourier_power_spectrum']['mean']
            self.mss.append(np.sum(((pwr - mean) / mean) ** 2) / (1.0 * (np.size(pwr))))


