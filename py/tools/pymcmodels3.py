"""
PyMC models for the red noise study.  These models have a Jeffreys' prior for
the noise in the data.

All the models have two objects in common:

(1) a function called "fourier_power_spectrum" which is the theoretical
    description of the observed spectrum,
(2) an object "spectrum" which is likelihood of the observed spectrum given
    the model.

Notes
-----

"""


import pymc
import rnspectralmodels

# -----------------------------------------------------------------------------
# Power law with a constant
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

    # Factor that expresses our uncertainty over the number of independent
    # pixels that comprise the observation
    nfactor = pymc.OneOverX('nfactor', value=1.0)

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
                           tau=1.0 / ((sigma ** 2) / nfactor),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / ((sigma ** 2) / nfactor),
                             mu=fourier_power_spectrum)
    # MCMC model
    return locals()

# -----------------------------------------------------------------------------
# Power law with a constant, plus a Gaussian shaped bump
#
def Log_splwc_AddNormalBump2(analysis_frequencies, analysis_power, sigma,
                             init=None,
                             log_bump_frequency_limits=[2.0, 6.0]):
    """
    Power law with a constant, plus a Gaussian shaped bump.
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

    # Factor that expresses our uncertainty over the number of independent
    # pixels that comprise the observation
    nfactor = pymc.OneOverX('nfactor', value=1.0)

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
        out = rnspectralmodels.Log_splwc_AddNormalBump2(f, [a, p, b, ga, gc, gs])
        return out

    spectrum = pymc.Normal('spectrum',
                           tau=1.0 / ((sigma ** 2) / nfactor),
                           mu=fourier_power_spectrum,
                           value=analysis_power,
                           observed=True)

    predictive = pymc.Normal('predictive',
                             tau=1.0 / ((sigma ** 2) / nfactor),
                             mu=fourier_power_spectrum)

    # MCMC model
    return locals()

