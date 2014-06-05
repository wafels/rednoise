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
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=-20.0,
                                      upper=10.0,
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
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      lower=-20.0,
                                      upper=5.0,
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      lower=log_bump_frequency_limits[0],
                                      upper=log_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      lower=0.001,
                                      upper=3.0,
                                      doc='gaussian_width')
    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      value=init[3],
                                      lower=-20.0,
                                      upper=5.0,
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      value=init[4],
                                      lower=log_bump_frequency_limits[0],
                                      upper=log_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      value=init[5],
                                      lower=0.001,
                                      upper=3.0,
                                      doc='gaussian_width')

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


# -----------------------------------------------------------------------------
# Model: np.log( power law plus constant + exponential decay autocorrelation )
#
def Log_splwc_AddExpDecayAutoCor(analysis_frequencies, analysis_power, sigma,
                             init=None,
                             normalized_frequency_limits=[2.0, 1000.0]):
    """
    Model: np.log( power law plus constant + exponential decay autocorrelation )
    """
    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        eda_amplitude = pymc.Uniform('eda_amplitude',
                                      lower=-20.0,
                                      upper=0.0,
                                      doc='eda_amplitude')

        eda_frequency = pymc.Uniform('eda_frequency',
                                      lower=normalized_frequency_limits[0],
                                      upper=normalized_frequency_limits[1],
                                      doc='eda_frequency')

        eda_index = pymc.Uniform('eda_index',
                                      lower=0.001,
                                      upper=6.0,
                                      doc='eda_width')
    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[0],
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[1],
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        eda_amplitude = pymc.Uniform('eda_amplitude',
                                      value=init[3],
                                      lower=-20.0,
                                      upper=0.0,
                                      doc='eda_amplitude')

        eda_frequency = pymc.Uniform('eda_frequency',
                                      value=init[4],
                                      lower=normalized_frequency_limits[0],
                                      upper=normalized_frequency_limits[1],
                                      doc='eda_frequency')

        eda_index = pymc.Uniform('eda_index',
                                      value=init[5],
                                      lower=0.001,
                                      upper=6.0,
                                      doc='eda_width')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ea=eda_amplitude,
                               ep=eda_frequency,
                               ei=eda_index,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.Log_splwc_AddExpDecayAutocor(f, [a, p, b, ea, ep, ei])
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
# Model : np.log( broken power law plus constant )
#
def Log_double_broken_power_law_with_constant(analysis_frequencies, analysis_power, sigma,
                                              init=None):
    """
    Model : np.log( broken power law plus constant )
    """
    if init == None:
        power_law_index1 = pymc.Uniform('power_law_index1',
                                       lower=0.0001,
                                       upper=6.0,
                                       doc='power law index 1')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        break_frequency = pymc.Uniform('break_frequency',
                                      lower=1.0,
                                      upper=7.0,
                                      doc='break frequency')

        power_law_index2 = pymc.Uniform('power_law_index2',
                                       lower=0.0001,
                                       upper=6.0,
                                       doc='power law index 2')

    else:
        power_law_index1 = pymc.Uniform('power_law_index1',
                                       value=init[0],
                                       lower=0.0001,
                                       upper=6.0,
                                       doc='power law index 1')

        power_law_norm = pymc.Uniform('power_law_norm',
                                       value=init[1],
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                       value=init[2],
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        break_frequency = pymc.Uniform('break_frequency',
                                       value=init[3],
                                      lower=-20.0,
                                      upper=0.0,
                                      doc='break frequency')

        power_law_index2 = pymc.Uniform('power_law_index2',
                                       value=init[4],
                                       lower=0.0001,
                                       upper=6.0,
                                       doc='power law index 2')

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(n1=power_law_index1,
                               a=power_law_norm,
                               b=background,
                               brk=break_frequency,
                               n2=power_law_index2,
                               f=analysis_frequencies):
        #A pure and simple power law model#
        out = rnspectralmodels.Log_double_broken_power_law_with_constant(f, [a, n1, b, brk, n2])
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
# Model : np.log( power law plus constant + normal bump )
#
def Log_splwc_AddNormalBump2(analysis_frequencies, analysis_power, sigma,
                             init=None,
                             normalized_bump_frequency_limits=[1.0, 1000.0],
                             sigma_type=None):
    """
    Model : np.log( power law plus constant + normal bump )
    """
    if init == None:
        power_law_index = pymc.Uniform('power_law_index',
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      lower=-20.0,
                                      upper=5.0,
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      lower=normalized_bump_frequency_limits[0],
                                      upper=normalized_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      lower=0.001,
                                      upper=3.0,
                                      doc='gaussian_width')
    else:
        power_law_index = pymc.Uniform('power_law_index',
                                       value=init[1],
                                       lower=-1.0,
                                       upper=6.0,
                                       doc='power law index')

        power_law_norm = pymc.Uniform('power_law_norm',
                                      value=init[0],
                                      lower=-10.0,
                                      upper=10.0,
                                      doc='power law normalization')

        background = pymc.Uniform('background',
                                      value=init[2],
                                      lower=-20.0,
                                      upper=10.0,
                                      doc='background')

        gaussian_amplitude = pymc.Uniform('gaussian_amplitude',
                                      value=init[3],
                                      lower=-20.0,
                                      upper=5.0,
                                      doc='gaussian_amplitude')

        gaussian_position = pymc.Uniform('gaussian_position',
                                      value=init[4],
                                      lower=normalized_bump_frequency_limits[0],
                                      upper=normalized_bump_frequency_limits[1],
                                      doc='gaussian_position')

        gaussian_width = pymc.Uniform('gaussian_width',
                                      value=init[5],
                                      lower=0.001,
                                      upper=3.0,
                                      doc='gaussian_width')

    # Putting a prior on the noise estimation
    if sigma_type is not None:
        type = sigma_type["type"]
        # Jeffreys' prior
        if type == 'jeffreys':
            sfactor = pymc.OneOverX('sfactor', value=1.0, doc='sfactor')
            nfactor = sfactor * sigma_type["npixels"]
        # Prior estimated from the shape of the cross-correlation distribution
        if type == 'beta':
            if init == None:
                sfactor = pymc.Beta('sfactor',
                                    sigma_type['alpha'], sigma_type['beta'])
            else:
                sfactor = pymc.Beta('sfactor',
                                    sigma_type['alpha'], sigma_type['beta'],
                                    value=init[6])
            nfactor = 1.0 + (sigma_type["npixels"] - 1) * sfactor
    else:
        nfactor = 1.0

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def fourier_power_spectrum(p=power_law_index,
                               a=power_law_norm,
                               b=background,
                               ga=gaussian_amplitude,
                               gc=gaussian_position,
                               gs=gaussian_width,
                               f=analysis_frequencies):

        out = rnspectralmodels.Log_splwc_AddNormalBump2(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / (sigma ** 2) / nfactor,
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / (sigma ** 2) / nfactor,
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()
