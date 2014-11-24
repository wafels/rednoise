"""
PyMC models for the red noise study.

All the models have two objects in common:

(1) a function called "fourier_power_spectrum" which is the theoretical
    description of the observed spectrum,
(2) an object "spectrum" which is likelihood of the observed spectrum given
    the model.
Notes
-----

"""


import numpy as np
import pymc
import rnspectralmodels
from scipy.stats.kde import gaussian_kde


# Limits on the fit
limits = {"power_law_index": [1.0, 6.0],
          "power_law_norm": [-10.0, np.log(1000.0)],
          "background": [-200.0, np.log(100.0)]}

glimits = {"gaussian_amplitude": [-20.0*1000000, limits["power_law_norm"][1]],
          "gaussian_width": [0.001, 3.00]}


#
# Get a smooth pdf from some input data
#
def calculate_kde(dataset, bw_method):
    return gaussian_kde(dataset, bw_method)


def KernelSmoothing(name, dataset, bw_method=None, lower=-np.inf, upper=np.inf, observed=False, value=None):
    '''Create a pymc node whose distribution comes from a kernel smoothing density estimate.'''
    density = calculate_kde(dataset, bw_method)
    lower_tail = 0
    upper_tail = 0
    if lower > -np.inf:
        lower_tail = density.integrate_box(-np.inf, lower)
    if upper < np.inf:
        upper_tail = density.integrate_box(upper, np.inf)
    factor = 1.0 / (1.0 - lower_tail - upper_tail)

    def logp(value):
        if value < lower or value > upper:
            return -np.inf
        d = density(value)
        if d == 0.0:
            return -np.inf
        return np.log(factor * density(value))

    def random():
        result = None
        while result == None:
            result = density.resample(1)[0][0]
            if result < lower or result > upper:
                result = None
        return result

    if value == None:
        value = random()

    dtype = type(value)

    result = pymc.Stochastic(logp=logp,
                             doc='A kernel smoothing density node.',
                             name=name,
                             parents={},
                             random=random,
                             trace=True,
                             value=dataset[0],
                             dtype=dtype,
                             observed=observed,
                             cache_depth=2,
                             plot=True,
                             verbose=0)
    return result


def splwc_exp(analysis_frequencies, analysis_power, init=None):
    """Power law with a constant.  This model assumes that the power
    spectrum is made up of a power law and a constant background.  At high
    frequencies the power spectrum is dominated by the constant background.
    """
    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.power_law_with_constant(f, [a, p, b])
        return out

    predictive = pymc.Exponential('predictive',
                           beta = 1.0 / fourier_power_spectrum)

    spectrum = pymc.Exponential('spectrum',
                           beta = 1.0 / fourier_power_spectrum,
                           value=analysis_power, observed=True)

    # MCMC model
    return locals()


# -----------------------------------------------------------------------------
# Model : np.log( power law plus constant )
#
def Log_splwc(analysis_frequencies, analysis_power, sigma, init=None):
    """Power law with a constant.  This model assumes that the power
    spectrum is made up of a power law and a constant background.  At high
    frequencies the power spectrum is dominated by the constant background.
    """
    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.Log_splwc(f, [a, p, b])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / (sigma ** 2),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / (sigma ** 2),
                             mu=fourier_power_spectrum)
    # MCMC model
    return locals()


# -----------------------------------------------------------------------------
# Model : np.log( power law plus constant )
#
def Log_splwc_noise_prior(analysis_frequencies, analysis_power, sigma, npx, init=None):
    """Power law with a constant.  This model assumes that the power
    spectrum is made up of a power law and a constant background.  At high
    frequencies the power spectrum is dominated by the constant background.
    """
    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')
    if np.isscalar(npx):
        # Only the number of pixels counts
        npixel = pymc.Uniform('npixel',
                                      lower=1.0,
                                      upper=npx,
                                      doc='npixel')
    else:
        # Use Kernel Density Estimation to create an estimated PDF for the
        # number of pixels
        print 'Using KDE estimate of independent pixels'
        npixel = KernelSmoothing('KDE_estimate', npx['npixel_model'], lower=1, upper=npx['npixels'])

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.Log_splwc(f, [a, p, b])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / (sigma ** 2) / (1.0 * npixel),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / (sigma ** 2) / (1.0 * npixel),
                             mu=fourier_power_spectrum)
    # MCMC model
    return locals()


# -----------------------------------------------------------------------------
# Model: np.log( power law plus constant + lognormal )
#
def Log_splwc_AddLognormalBump2_noise_prior(analysis_frequencies, analysis_power, sigma, npx,
                             init=None,
                             log_bump_frequency_limits=[2.0, 6.0]):
    """
    Model: np.log( power law plus constant + lognormal )
    """
    if np.isscalar(npx):
        # Only the number of pixels in the area is used
        npixel = pymc.Uniform('npixel',
                                      lower=1.0,
                                      upper=npx,
                                      doc='npixel')
    else:
        # Use Kernel Density Estimation to create an estimated PDF for the
        # number of independent pixels
        print 'Using KDE estimate of independent pixels'
        npixel = KernelSmoothing('KDE_estimate', npx['npixel_model'], lower=1, upper=npx['npixels'])

    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      lower=glimits['gaussian_amplitude'][0],
                                      upper=glimits['gaussian_amplitude'][1],
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      lower=log_bump_frequency_limits[0],
                                      upper=log_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      lower=glimits['gaussian_width'][0],
                                      upper=glimits['gaussian_width'][1],
                                      doc='gaussian_width')

    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      value=init[3],
                                      lower=glimits['gaussian_amplitude'][0],
                                      upper=glimits['gaussian_amplitude'][1],
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      value=init[4],
                                      lower=log_bump_frequency_limits[0],
                                      upper=log_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      value=init[5],
                                      lower=glimits['gaussian_width'][0],
                                      upper=glimits['gaussian_width'][1],
                                      doc='gaussian_width')

    # Set a bound on the Gaussian Amplitude
    @pymc.potential
    def gaussian_amplitude_bound(gaussian_amplitude=gaussian_amplitude,
                                 power_law_norm=power_law_norm):
        if gaussian_amplitude <= power_law_norm:
            return 0.0
        else:
            return -np.inf

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ga=gaussian_amplitude,
                               gc=gaussian_position,
                               gs=gaussian_width,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.Log_splwc_AddLognormalBump2(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / (sigma ** 2) / (1.0 * npixel),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / (sigma ** 2) / (1.0 * npixel),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()


# -----------------------------------------------------------------------------
# Model: np.log( power law plus constant + lognormal )
#
def Log_splwc_AddLognormalBump2(analysis_frequencies, analysis_power, sigma,
                             init=None,
                             log_bump_frequency_limits=[2.0, 6.0]):
    """
    Model: np.log( power law plus constant + lognormal )
    """

    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      lower=glimits['gaussian_amplitude'][0],
                                      upper=glimits['gaussian_amplitude'][1],
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      lower=log_bump_frequency_limits[0],
                                      upper=log_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      lower=glimits['gaussian_width'][0],
                                      upper=glimits['gaussian_width'][1],
                                      doc='gaussian_width')

    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=limits['power_law_index'][0],
                                       upper=limits['power_law_index'][1],
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=limits['power_law_norm'][0],
                                      upper=limits['power_law_norm'][1],
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=limits['background'][0],
                                      upper=limits['background'][1],
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      value=init[3],
                                      lower=glimits['gaussian_amplitude'][0],
                                      upper=glimits['gaussian_amplitude'][1],
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      value=init[4],
                                      lower=log_bump_frequency_limits[0],
                                      upper=log_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      value=init[5],
                                      lower=glimits['gaussian_width'][0],
                                      upper=glimits['gaussian_width'][1],
                                      doc='gaussian_width')

    # Set a bound on the Gaussian Amplitude
    @pymc.potential
    def gaussian_amplitude_bound(gaussian_amplitude=gaussian_amplitude,
                                 power_law_norm=power_law_norm):
        if gaussian_amplitude < power_law_norm:
            return 0.0
        else:
            return -np.inf

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ga=gaussian_amplitude,
                               gc=gaussian_position,
                               gs=gaussian_width,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.Log_splwc_AddLognormalBump2(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / (sigma ** 2),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / (sigma ** 2),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()
