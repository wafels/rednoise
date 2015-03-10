import os
import numpy as np
from scipy.stats import expon
import rnspectralmodels
import lnlike_model_fit as lnlike_model_fit
import matplotlib.pyplot as plt

outputdir = os.path.expanduser('~/ts/img/test_lnlikefit_2')

#
# add a small amount of deviation in the spectrum.  The reduced chi-squared
# varies appropriately.
#

#
# set up some test data
#
indices = [1.2, 1.7, 2.2]
widths = [0.01, 0.1, 1.0]
pos_indices = [40, 100, 400]

for index in indices:
    for width in widths:
        for pos_index in pos_indices:

            # Number of frequencies
            nf = 899

            # Normalized frequencies
            freqs = 1.0 * np.arange(1, nf)

            # True frequencies
            df = 1.0 / (1800 * 12.0)
            truefreq = freqs * df

            # Power law values
            plaw_values = [np.log(1.0), index]

            # True model which will be used to create the power spectra
            true_model = rnspectralmodels.power_law_with_gaussian

            # Model that we will use to fit the data
            fit_model = rnspectralmodels.power_law

            # Gaussian position in normalized frequency units
            pos = np.log(1.0 * pos_index)

            # log10(amplitude range)
            amp_range = [-3.0, 3.0]

            # Number of amplitudes
            namp = 1000

            # Amplitude multiplier
            multiplier = 10.0 ** np.arange(amp_range[0], amp_range[1], (amp_range[1] - amp_range[0]) / np.float64(namp))

            # Storage for the reduced chi-squared results
            rchi2s = []
            ratio = []
            power_law_index = []
            amps = np.zeros_like(multiplier)
            for iamp in range(0, len(multiplier)):

                # Now set up some simulated data and do the fit
                # Underlying power law
                PL = fit_model(plaw_values, freqs)

                # Value of the underlying power at the location of the Gaussian
                value = PL[pos_index]

                # log amplitude of the Gaussian we will use
                amps[iamp] = np.log(multiplier[iamp] * value)

                # True model
                S = true_model([plaw_values[0], plaw_values[1], amps[iamp], pos, width], freqs)

                # Ratio of the True model to the under lying power law
                ratio.append(S[pos_index] / PL[pos_index])

                # Noisy sample data
                data = np.zeros_like(freqs)
                for i, s in enumerate(S):
                    data[i] = expon.rvs(scale=s)

                # Initial guess
                initial_guess = [np.log(data[0]), 1.5, np.log(data[-1])]

                # Do the fit
                result = lnlike_model_fit.go(freqs, data, fit_model, initial_guess, "Nelder-Mead")

                # Power law index
                power_law_index.append(result['x'][1])

                # Calculate Nita et al reduced chi-squared value
                bestfit = fit_model(result['x'], freqs)

                # Sample to model ratio
                rhoj = lnlike_model_fit.rhoj(data, bestfit)

                # Nita et al (2014) value of the reduced chi-squared
                rchi2 = lnlike_model_fit.rchi2(1, len(freqs) - len(initial_guess) - 1, rhoj)

                # Save them all
                rchi2s.append(rchi2)

            ratio_label = 'log(ratio of true model to true underlying power law at Gaussian position)'
            title = 'Gaussian center=%f Hz, log(width)=%f' % (truefreq[pos_index], width)
            particular = 'n%f_=c=%f_log(w)=%f' % (plaw_values[1], truefreq[pos_index], width)

            plt.close('all')
            plt.figure(1)
            plt.plot(amps, rchi2s)
            plt.axhline(1.0, color='k', label='reduced chi-squared=1')
            plt.legend(framealpha=0.5, loc=2)
            plt.xlabel('log(amplitude)')
            plt.ylabel('reduced chi-squared')
            plt.title(title)
            plt.savefig(os.path.join(outputdir, '%s.amplitude_vs_rchi2.png' % particular))

            plt.figure(2)
            plt.semilogy(amps, ratio, label='true model / fit model at Gaussian position')
            plt.legend(framealpha=0.5, loc=2)
            plt.xlabel('log(amplitude)')
            plt.ylabel(ratio_label)
            plt.title(title)
            plt.savefig(os.path.join(outputdir, '%s.amplitude_vs_ratio.png' % particular))

            plt.figure(3)
            plt.semilogx(ratio, rchi2s)
            plt.axhline(1.0, color='k', label='reduced chi-squared=1')
            plt.ylabel('reduced chi-squared')
            plt.xlabel(ratio_label)
            plt.title(title)
            plt.savefig(os.path.join(outputdir, '%s.ratio_vs_rchi2.png' % particular))

            pvalue = np.array([0.025, 0.975])
            rchi2limit = [lnlike_model_fit.rchi2_given_prob(pvalue[1], 1.0, nf - 2 - 1),
                          lnlike_model_fit.rchi2_given_prob(pvalue[0], 1.0, nf - 2 - 1)]
            plt.axhline(rchi2limit[0], color='k', linestyle=':', label='lower')
            plt.axhline(rchi2limit[1], color='k', linestyle=':', label='upper')
            plt.legend(framealpha=0.5, loc=2)
            plt.savefig(os.path.join(outputdir, '%s.ratio_vs_rchi2.png' % particular))

            plt.figure(4)
            plt.semilogx(ratio, power_law_index)
            plt.axhline(plaw_values[1], color='k', label='true power law index = %f' % plaw_values[1])
            plt.xlabel(ratio_label)
            plt.ylabel('power law index')
            plt.title(title)

            # Indicate the ones with good fits
            label_first_one = True
            for ipli in range(0, len(power_law_index)):
                if (rchi2s[ipli] >= rchi2limit[0]) and (rchi2s[ipli] <= rchi2limit[1]):
                    if label_first_one:
                        label = 'power law index within reduced chi-squared limits'
                        label_first_one = False
                    else:
                        label = None
                    plt.axvline(power_law_index[ipli], color='r', label=label, linestyle=':')
            plt.legend(framealpha=0.5, loc=2)
            plt.savefig(os.path.join(outputdir, '%s.ratio_vs_powerlawindex.png' % particular))

            plt.figure(5)
            plt.loglog(freqs, S, label='true model at maximum Gaussian amplitude')
            plt.loglog(freqs, data, label='simulated data')
            plt.loglog(freqs, bestfit, label='best fit')
            plt.xlabel('normalized frequencies')
            plt.ylabel('emission (arb units)')
            plt.title(title)
            plt.legend(framealpha=0.5, loc=3)
            plt.savefig(os.path.join(outputdir, '%s.maximum_disturbance_powerspectrum.png' % particular))

