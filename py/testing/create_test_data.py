import numpy as np
from timeseries import TimeSeries
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
#import pickle


def create_test_data():

    dt = 12.0
    nt = 300
    np.random.seed(seed=1)
    pls1 = SimplePowerLawSpectrumWithConstantBackground([10.0, 2.0, -5.0],
                                                        nt=nt,
                                                        dt=dt)
    data = TimeSeriesFromPowerSpectrum(pls1).sample
    t = dt * np.arange(0, nt)
    amplitude = 0.0
    data = data + amplitude * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)
    ts = TimeSeries(t, data)
    return ts
