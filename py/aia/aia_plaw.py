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


# A Gaussian shape
def GaussianShape(x, a, xc, sigma):
    #constant = (1.0 / np.sqrt(2.0 * np.pi))
    z = (x - xc) / sigma
    return a * np.exp(-0.5 * z ** 2)


# Do the fit
def do_fit(freqs, pwrinput, func, guessfunc=None, p0=None, sigma=None):
    """
    Fit an arbitrary function, starting from a guess function that has also
    been supplied.  Fit over all the values in the input pwrinput.
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
            if guessfunc is not None:
                pguess = guessfunc(y, p0)
            else:
                pguess = [y[0], 1.8, y[-1]]

            # Do the fit
            try:
                results = curve_fit(func, freqs, y, p0=pguess, sigma=sigma)
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
