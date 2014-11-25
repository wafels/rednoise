__author__ = 'ireland'

import numpy as np
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from timeseries import TimeSeries


def fakedata():
    print 'Generating fake data'
    dt = 12.0
    nt = 300
    np.random.seed(seed=1)
    model_param = [10.0, 1.77, -100000.0]
    pls1 = SimplePowerLawSpectrumWithConstantBackground(model_param,
                                                        nt=nt,
                                                        dt=dt)
    data = TimeSeriesFromPowerSpectrum(pls1).sample
    t = dt * np.arange(0, nt)
    amplitude = 0.0
    data = data + amplitude * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)

    # Linear increase
    lin = 0.0
    data = data + lin * t / np.max(t)

    # constant
    constant = 10.0
    data = data + constant

    return t, data, model_param