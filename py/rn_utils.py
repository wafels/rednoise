"""
Utility functions for the 1/f study
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
from scipy.stats import chi2, uniform

class PowerLawPowerSpectrum():
    def __init__(self, frequencies=None, power=None, parameters=None):
        self.frequencies = frequencies
        self.power = power
        self.parameters = parameters
        self.logpower = np.log(self.power)

class SimplePowerLawSpectrum(PowerLawPowerSpectrum):
    def __init__(self):
        self.power = self.parameters[0] * self.frequencies ** self.parameters[1]

class SimplePowerLawSpectrumWithConstantBackground(PowerLawPowerSpectrum):
    def __init__(self):
        self.power = self.parameters[0] * self.frequencies ** self.parameters[1] + self.parameters[2]


def power_law_power_spectrum_time_series(f, alpha, norm):
    """Create a time series with a power law power spectrum"""

    fft_sim = np.append(np.zeros((1)), f ** (-alpha / 2.0))

    T_sim = norm * np.fft.irfft(fft_sim)
    return T_sim


def power_law_noise(n, dt, alpha, seed=None):
    """Create a time series with power law noise"""

    # White noise
    np.random.seed(seed=seed)
    wn = np.random.normal(size=(n))

    # FFT of the white noise - chi2(2) distribution
    wn_fft = np.fft.rfft(wn)

    # frequencies
    f = np.fft.fftfreq(n, dt)[:len(wn_fft)]
    f[-1] = np.abs(f[-1])

    fft_sim = wn_fft[1:] * f[1:] ** (-alpha / 2.0)
    T_sim = np.fft.irfft(fft_sim)
    return T_sim

def power_law_time_series(V, W, N, dt):
    """
    Create a time series following the recipe of Vaughan (2010), MNRAS, 402,
    307, appendix B
    
    Parameters
    ----------
    
    """


def power_law_noise_random_phases(S, fft_zero=0.0, seed=None):
    """Create a time series with power law noise where Following the recipe
    of Vaughan (2010), MNRAS, 402, 307, appendix B
    
    Parameters
    ----------
    S : numpy array
        Theoretical fourier power spectrum, from the first non-zero frequency
        up to the Nyquist frequency.
 
    fft_zero : scalar number
        The value at the zero Fourier frequency.

    seed : scalar number
        A seed value for the random number generator
     
    """
    
    # Number of frequencies to calculate
    K = len(S)

    # chi-squared(2) random numbers for all frequencies except the nyquist
    # frequency
    X = np.concatenate((chi2.rvs(2, size=K-1, seed=seed),
                        chi2.rvs(1, size=1, seed=seed)))
    
    # random phases, except for the nyquist frequency.
    ph = uniform.rvs(loc=-np.pi/2.0, scale=np.pi, size=K, seed=seed)
    ph[-1] = 0.0
    
    # power spectrum
    I = S*X/2.0
    
    # Amplitudes
    A = np.sqrt(I/2.0)
    
    # Complex vector
    F = A*np.exp(-np.complex(0,1)*ph)
    
    # Form the negative frequency part
    F_negative = np.conjugate(F)[::-1]
    
    # Form the fourier transform
    F_complete = np.concatenate((np.asarray([fft_zero]), F, F_negative))
    
    # create the time-series.  The complex part should be tiny.
    T_sim = np.fft.ifft(F_complete)
    
    # The time series is formally complex.  Return the real part only.
    return np.real(T_sim)


def simulated_power_law(n, dt, alpha, n_oversample=10, dt_oversample=10,
                        seed=None, minimum=None, poisson=False, amplitude=1.0):
    """Create a time series of length n and sample size dt"""
    data = power_law_noise(n_oversample * n, dt / dt_oversample, alpha,
                           seed=seed)
    data = data[0.5 * len(data) - n / 2: 0.5 * len(data) + n / 2]

    if minimum is not None:
        data = amplitude*(data - np.min(data)) + minimum

    if poisson:
        data = np.random.poisson(lam=data)

    return data


def credible_interval(data, ci=0.68):
    """find the limits where the upper 100*(1-ci/2)% of the data, and the lower
    100*ci/2 % of the data are"""
    s = np.sort(data)
    ns = s.size
    for i in range(0, ns):
        fraction = np.count_nonzero(s[i] > s) / (1.0 * ns)
        if fraction >= (1.0 - ci) / 2.0:
            break
    lower_limit = s[i]

    for i in range(ns - 1, 0, -1):
        fraction = np.count_nonzero(s[i] > s) / (1.0 * ns)
        if fraction <= 1.0 - (1.0 - ci) / 2.0:
            break
    upper_limit = s[i]
    return np.asarray([lower_limit, upper_limit])


def do_simple_fit(freq, power, show=False):
    """Do a very simple fit to the frequencies and power"""
    x = np.log(freq)
    y = np.log(power)
    coefficients = np.polyfit(x, y, 1)
    m_estimate = -coefficients[0]
    c_estimate = np.exp(coefficients[1])

    if show:
        # Fit to the power spectrum
        power_fit = c_estimate * freq ** (-m_estimate)

        # plot the power spectrum and the quick fit
        plt.figure(2)
        plt.loglog(freq, power, label='observed power')
        plt.loglog(freq, power_fit, label='fit, power law index: ' + str(m_estimate))
        plt.xlabel('frequency')
        plt.ylabel('power')
        plt.title('Data fit with simple single power law')
        plt.legend()
        plt.show()
    return [c_estimate, m_estimate]


def plot_ts_duration_and_power_law_index_results(pickle_directory, filename, img_directory, format):
    # Load in the data
    pkl_file = open(pickle_directory + filename + '.pickle', 'rb')
    results = pickle.load(pkl_file)
    nkeep = results["nkeep"]
    fraction_found_ci = results["fraction_found_ci"]
    fraction_found_mean = results["fraction_found_mean"]
    fraction_found_mode = results["fraction_found_mode"]
    ntrial = results["bayes_mode"].shape[1]

    alpha = results["alpha"]
    if "alpha_range" in results:
        alpha_range = results["alpha_range"]
    else:
        alpha_range = 0.1

    plt.semilogx(nkeep, fraction_found_ci,
                 label=r'$[\alpha_{68}^{L},\alpha_{68}^{H}] \in [\alpha_{true}- %3.1f, \alpha_{true}+ %3.1f]$' % (alpha_range, alpha_range))
    plt.semilogx(nkeep, fraction_found_mean,
                 label=r'$\overline{\alpha}\in [\alpha_{true}- %3.1f, \alpha_{true}+ %3.1f]$' % (alpha_range, alpha_range))
    plt.semilogx(nkeep, fraction_found_mode,
                 label=r'$\alpha_{mode}\in [\alpha_{true}- %3.1f, \alpha_{true}+ %3.1f]$' % (alpha_range, alpha_range))

    plt.xlabel("length of time series (# samples)")
    plt.ylabel("fraction found (# trials=%8i)" % (ntrial))
    plt.title(r"Fraction found with $\alpha=%3.1f$" % (alpha))
    plt.legend(loc=4)
    plt.savefig(img_directory + filename, format=format)
    return


def write_ts_as_csv(csv_directory, filename, t, data):
    """Write the time-series data out as a CSV file"""
    # open output file
    outfile = open(csv_directory + filename + '.csv', "wb")

    # get a csv writer
    writer = csv.writer(outfile)

    # write header
    writer.writerow(['sample time', 'data'])

    # write data
    for i in range(0, len(data)):
        writer.writerow([t[i], data[i]])

    # close file
    outfile.close()
    return


def bayes_mode(data, bins, precision):
    for k in range(0, 100):
        h, bin_edges = np.histogram(data, bins=bins * 2 ** k)
        if bin_edges[1] - bin_edges[0] <= 0.5 * precision:
            break
    return bin_edges[h.argmax()]


def summary_stats(data, precision, bins=40):
    """Summary statistics for an input array"""
    mean = np.mean(data)
    mode = bayes_mode(data, bins, precision)
    ci68 = credible_interval(data, ci=0.68)
    return {"mean": mean, "mode": mode, "ci68": ci68}
