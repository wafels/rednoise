"""
Fit functions used in the AIA power law work
"""
import numpy as np
from scipy.optimize import curve_fit


# A power law function
def PowerLawPlusConstant(freq, a, n, c):
    # The amplitude definition assures positivity
    return a * a * freq ** -n + c


# Log of the power spectrum model
def LogPowerLawPlusConstant(freq, a, n, c):
    return np.log(PowerLawPlusConstant(freq, a, n, c))


# A power law function with a Gaussian bump
def PowerLawPlusConstantGaussian(freq, a, n, c, ga, gc, gsigma):
    return PowerLawPlusConstant(freq, a, n, c) + GaussianShape(freq, ga, gc, gsigma)


# Log of the power law with bump
def LogPowerLawPlusConstantGaussian(freq, a, n, c, ga, gc, gsigma):
    return np.log(PowerLawPlusConstant(freq, a, n, c) + GaussianShape(freq, ga, gc, gsigma))


def fit_PowerLawPlusConstant(freqs, pwrinput, sigma=None):
    """
    Assumes that the data consists of a power law with a constant.
    """
    if pwrinput.ndim == 1:
        pwr = np.zeros((1, 1, pwrinput.size))
        pwr[0, 0, :] = pwrinput
    else:
        pwr = pwrinput

    ny = pwr.shape[0]
    nx = pwr.shape[1]

    # Answer array - 3 variables for the power law with a constant
    answer = np.zeros((ny, nx, 3))
    error = np.zeros((ny, nx, 3))

    for i in range(0, nx):
        for j in range(0, ny):
            # Data
            y = pwr[j, i, :]

            # Guess
            pguess = [y[0], 1.8, y[-1]]

            # Do the fit
            try:
                results = curve_fit(PowerLawPlusConstant, freqs, y, p0=pguess, sigma=sigma)
            except RuntimeError:
                # Fit cannot be found
                answer[j, i, :] = np.inf
                error[j, i, :] = np.inf
            else:
                # If the error array is messed up, store nans
                if np.any(np.isfinite(results[1]) == False):
                    answer[j, i, :] = np.nan
                    error[j, i, :] = np.nan
                else:
                    # Keep the results
                    answer[j, i, :] = results[0]
                    error[j, i, :] = np.sqrt(np.diag(results[1]))
    return answer, error


def BrokenLinear(x, m, c, x0):
    y = np.zeros_like(x)
    y[x <= x0] = m * x[x <= x0] + c
    y[x > x0] = m * x0 + c
    return y


def GaussianShape(x, a, xc, sigma):
    #constant = (1.0 / np.sqrt(2.0 * np.pi))
    z = (x - xc) / sigma
    return a * np.exp(-0.5 * z ** 2)


def ObserverdPowerSpectrumWithBump(x, m, c, x0, a, xc, sigma):
    ps = BrokenLinear(x, m, c, x0)
    return ps + GaussianShape(x, a, xc, sigma)


# Go through all the time series and generate a simple spectral fits
def fit_using_simple(freqs, pwr, sigma=None):
    """
    Simple fit to the data
    """
    ny = pwr.shape[0]
    nx = pwr.shape[1]

    # Answer array - 3 variables for the spectrum
    answer = np.zeros((ny, nx, 3))
    error = np.zeros((ny, nx, 3))

    for i in range(0, nx):
        for j in range(0, ny):
            y = np.log10(pwr[j, i, :])
            # Generate a guess for the power spectrum
            pguess = [y[0], 1.8, y[-1]]
            # Do the fit
            try:
                results = curve_fit(LogPowerLawPlusConstant,
                                freqs, y, p0=pguess, sigma=None)
            except RuntimeError:
                # Fit cannot be found
                answer[j, i, :] = np.nan
                error[j, i, :] = np.nan
            else:
                # If the error array is messed up, store nans
                if np.any(np.isfinite(results[1]) == False):
                    answer[j, i, :] = np.nan
                    error[j, i, :] = np.nan
                else:
                    # Keep the results
                    answer[j, i, :] = results[0]
                    error[j, i, :] = np.sqrt(np.diag(results[1]))
    return answer, error


# Go through all the time series and generate a fit with a bump
def fit_using_bump(freqs, pwr):
    """
    Assumes that the data consists of a linear power law with
    a break and a bump of excess power.
    """
    x = np.log10(freqs)
    ny = pwr.shape[0]
    nx = pwr.shape[1]

    # Broken linear guesses
    x0guess = -1.75

    # Center of the Gaussian bump guess
    xc_guess = np.log10(1.0 / 300.0)

    # Answer array - 3 variables for the background spectrum
    # and 3 for the bump
    answer = np.zeros((ny, nx, 6))
    error = np.zeros((ny, nx, 6))

    for i in range(0, nx):
        for j in range(0, ny):
            y = np.log10(pwr[j, i, :])
            # Generate a guess for the broken linear
            mguess = (y[-1] - y[0]) / (x[-1] - x[0])
            cguess = y[0] - mguess * x[0]

            # Guess for the bump
            gauss_guess = [1.0, xc_guess, 0.25]

            # Final Guess
            p0 = [mguess, cguess, x0guess,
                  gauss_guess[0], gauss_guess[1], gauss_guess[2]]

            # Do the fit
            try:
                results = curve_fit(ObserverdPowerSpectrumWithBump,
                                x, y, p0=p0)
            except RuntimeError:
                # Fit cannot be found
                answer[j, i, :] = np.nan
                error[j, i, :] = np.nan
            else:
                # If the error array is messed up, store nans
                if np.any(np.isfinite(results[1]) == False):
                    answer[j, i, :] = np.nan
                    error[j, i, :] = np.nan
                else:
                    # Keep the results
                    answer[j, i, :] = results[0]
                    error[j, i, :] = np.sqrt(np.diag(results[1]))
    return answer, error