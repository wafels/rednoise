"""
The purpose of this program is analyze test data so we can find out how long
a time series has to be in order for it to have enough information in it that
we are able to determine its power law index a reasonable percentage of the
time.

Ha! Longer time series will lead to more precise determinations of the
power law index, but the proportion lying within the credible interval
should be approximately the same, and in the limit, identical. I think.

"""
 
import numpy as np
import matplotlib.pyplot as plt
import rn_utils, pymcmodels

import pymc
import pickle, os
from matplotlib import rc

# Use LaTeX in matplotlib - very nice.
rc('text', usetex=True)
plt.ioff()

# Dump plots?
show_plot = True

# Save to CSV?
save_to_csv = True

# Where to dump the pickle data
pickle_directory = os.path.expanduser('~/ts/pickle/')

# Where to put the images
img_directory = os.path.expanduser('~/ts/img/')

# Where to dump the CSV files
csv_directory = os.path.expanduser('~/ts/csv/')

# Set up the simulated data
# Initial length of the time series
n_initial = 600

# Sampling (SDO AIA sample rate is once every 12 seconds)
dt = 1.0

# power law index
alpha = 2.0

# Alpha range - this is the half width of a range.  Three measures of
# how close we are getting to the true value are calculated
# (1) The 68% CI lies within alpha-alpha_range -> alpha+alpha_range
# (2) The mean value of alpha found lies within
#     alpha-alpha_range -> alpha+alpha_range
# (3) The mode value of alpha found lies within
#     alpha-alpha_range -> alpha+alpha_range

alpha_range = 0.1

# The time-series is multiplied by this factor
n_increment = np.sqrt(2.0)

# Maximum number of increments
max_increment = 9

# Number of trial data runs at each time-series length
ntrial = 100

# PyMC variables
iterations = 50000
burn = iterations / 4
thin = 10

# storage array for the credible interval
ci_keep68 = np.zeros((max_increment, ntrial, 2))

# Least squares fit found
lstsqr_fit = np.zeros((max_increment, ntrial, 2))

# Bayesian posterior means
bayes_mean = np.zeros((max_increment, ntrial, 2))

# Bayesian posterior modes
bayes_mode = np.zeros((max_increment, ntrial, 2))

# Storage array for the fraction found
fraction_found_ci = np.zeros((max_increment))
fraction_found_mean = np.zeros((max_increment))
fraction_found_mode = np.zeros((max_increment))

# Storage array for the length of the time-series
nkeep = np.zeros((max_increment))

# Strings required to save data
alpha_S = str(np.around(alpha, decimals=2))
ar_S = str(np.around(alpha_range, decimals=2))

# Calculate a filename
filename = "ts_duration_and_power_law_index_" + alpha_S

# Histogram bin size
bins = 40

# Main loop
for i in range(0, max_increment):
    # increase the length of the time series
    n = np.int(np.rint(n_initial * (n_increment ** i)))
    nkeep[i] = n
    print 'Increment counter ', i, max_increment, n

    for j in range(0, ntrial):
        print 'Trial counter ', j, ntrial
        # ensure repeatable random time-series.  We will need this when we
        # come to analyze the same data in different ways
        seed = j + i * ntrial
        test_data = rn_utils.simulated_power_law(n, dt, alpha, seed=seed)

        # Save to CSV file if required
        if save_to_csv:
            rn_utils.write_ts_as_csv(csv_directory,
                                     filename + '_seed=' + str(seed),
                                     dt * np.arange(0, len(test_data)),
                                     test_data)

        # get the power spectrum and frequencies we will analyze at
        observed_power_spectrum = (np.absolute(np.fft.fft(test_data))) ** 2
        fftfreq = np.fft.fftfreq(n, dt)
        these_frequencies = fftfreq > 0
        analysis_frequencies = fftfreq[these_frequencies[:-1]]
        analysis_power = observed_power_spectrum[these_frequencies[:-1]]

        # get a simple estimate of the power law index
        estimate = rn_utils.do_simple_fit(analysis_frequencies, analysis_power)
        c_estimate = estimate[0]
        m_estimate = estimate[1]

        # Keep the least squares fit
        lstsqr_fit[i, j, 0] = c_estimate
        lstsqr_fit[i, j, 1] = m_estimate

        # plot the power law spectrum
        power_fit = c_estimate * analysis_frequencies ** (-m_estimate)
        
        # Define the model we are going to use
        single_power_law_model = pymcmodels.single_power_law(analysis_frequencies, analysis_power,
                                c_estimate, m_estimate)

        # Set up the MCMC model
        M1 = pymc.MCMC(single_power_law_model)

        # Run the sampler
        M1.sample(iter=iterations, burn=burn, thin=thin, progress_bar=False)

        # Get the power law index and the normalization
        pli = M1.trace("power_law_index")[:]
        bayes_mean[i, j, 1] = np.mean(pli)
        cli = M1.trace("power_law_norm")[:]
        bayes_mean[i, j, 0] = np.mean(cli)

        # Get various properties of the power law index marginal distribution
        # and save them
        ci_keep68[i, j, :] = rn_utils.credible_interval(pli, ci=0.68)

        for k in range(0, 100):
            h, bin_edges = np.histogram(pli, bins=bins * 2 ** k)
            if bin_edges[1] - bin_edges[0] <= 0.5 * alpha_range:
                break
        bayes_mode[i, j, 1] = bin_edges[h.argmax()]

        for k in range(0, 100):
            h, bin_edges = np.histogram(cli, bins=bins * 2 ** k)
            if bin_edges[1] - bin_edges[0] <= 0.5 * alpha_range:
                break
        bayes_mode[i, j, 0] = bin_edges[h.argmax()]

        if show_plot:
            # Plots
            # Calculate the fit arising from the Bayes' mean
            bayes_mean_fit = bayes_mean[i, j, 0] * analysis_frequencies ** (-bayes_mean[i, j, 1])

            # Calculate the fit arising from the Bayes' mode
            bayes_mode_fit = bayes_mode[i, j, 0] * analysis_frequencies ** (-bayes_mode[i, j, 1])

            # plot the power spectrum, the quick fit, and the Bayesian fit
            plt.figure(1)
            plt.subplot(2, 2, 1)

            plt.plot(test_data)
            plt.xlabel('sample number (n=' + str(n) + ')')
            plt.ylabel('simulated emission')
            plt.title(r'Simulated data $\alpha_{true}$=%4.2f , seed=%8i ' % (alpha, seed))
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.loglog(analysis_frequencies, analysis_power, label=r'observed power: $\alpha_{true}= %4.2f$' % (alpha))
            plt.loglog(analysis_frequencies, power_fit, label=r'$\alpha_{lstsqr}=%4.2f$' % (m_estimate))
            plt.loglog(analysis_frequencies, bayes_mean_fit, label=r'$\overline{\alpha}=%4.2f$' % (bayes_mean[i, j, 1]))
            plt.loglog(analysis_frequencies, bayes_mode_fit, label=r'$\alpha_{mode}=%4.2f$' % (bayes_mode[i, j, 1]))
            plt.xlabel('frequency')
            plt.ylabel('power')
            plt.title('Data fit with simple single power law')
            plt.legend(loc=3)

            plt.subplot(2, 2, 3)
            plt.hist(pli, bins=bins, color='gray')
            plt.title('power law index PDF')
            plt.axvline(x=bayes_mean[i, j, 1], color='red', label=r'$\overline{\alpha}=%4.2f$' %(bayes_mean[i, j, 1]))
            plt.axvline(x=bayes_mode[i, j, 1], color='blue', label=r'$\alpha_{mode}=%4.2f$' %(bayes_mode[i, j, 1]))
            plt.axvline(x=ci_keep68[i, j, 0], color='green', label=r'$\alpha_{68}^{L}=%4.2f$' %(ci_keep68[i, j, 0]))
            plt.axvline(x=ci_keep68[i, j, 1], color='green', label=r'$\alpha_{68}^{H}=%4.2f$' %(ci_keep68[i, j, 1]))
            plt.axvline(x=alpha, color='black', label=r'$\alpha_{true}=%4.2f$' %(alpha))
            plt.axvline(x=m_estimate, color='black', linestyle='--', label=r'$\alpha_{lstsqr}=%4.2f$' %(m_estimate))
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.hist(cli, bins=bins)
            plt.title('power law normalization PDF')

            plt.show()

    # Calculate how often the credible interval is within a certain fixed range
    # of the true  value
    # (1) The 68% CI lies within alpha-alpha_range -> alpha+alpha_range
    low = np.squeeze(ci_keep68[i, :, 0]) >= alpha - alpha_range
    high = np.squeeze(ci_keep68[i, :, 1]) <= alpha + alpha_range
    fraction_found_ci[i] = np.sum(low * high) / (1.0 * ntrial)
    print fraction_found_ci

    # (2) The mean value of alpha found lies within
    #     alpha-alpha_range -> alpha+alpha_range
    high = np.squeeze(bayes_mean[i, :, 1]) <= alpha + alpha_range
    low = np.squeeze(bayes_mean[i, :, 1]) >= alpha - alpha_range
    fraction_found_mean[i] = np.sum(low * high) / (1.0 * ntrial)
    print fraction_found_mean

    # (3) The mode value of alpha found lies within
    #     alpha-alpha_range -> alpha+alpha_range
    high = bayes_mode[i, :] <= alpha + alpha_range
    low = bayes_mode[i, :] >= alpha - alpha_range
    fraction_found_mode[i] = np.sum(low * high) / (1.0 * ntrial)
    print fraction_found_mode

# Save the data to a pickle file
# Saving the 68% credible intervals, fraction found and length of time series
results = {"bayes_mean": bayes_mean,
           "bayes_mode": bayes_mode,
           "ci_keep68": ci_keep68,
           "fraction_found_ci": fraction_found_ci,
           "fraction_found_mean": fraction_found_mean,
           "fraction_found_mode": fraction_found_mode,
           "nkeep": nkeep, "alpha": alpha, "alpha_range": alpha_range}
pickle.dump(results, open(pickle_directory + filename + '.pickle', "wb"))

# Save a summary plot to file
rn_utils.plot_ts_duration_and_power_law_index_results(pickle_directory, filename, img_directory, 'png')

