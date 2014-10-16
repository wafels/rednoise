import numpy as np

imgdir = '/home/ireland/Documents/Proposals/2014/05 HSR Corona/Step 2/'


def norm_hanning(data):
    # Normalize
    data_mean = np.mean(data)
    data = (data - data_mean) / data_mean

    # length of the time series
    nt = len(data)

    # Multiply by the Hanning window
    return data * np.hanning(nt)


def equally_spaced(nt, duration):
    average_cadence = duration / np.float64(nt - 1)
    return average_cadence * np.arange(0, nt)
