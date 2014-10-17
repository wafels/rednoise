import numpy as np
from scipy.io import readsav

imgdir = '/home/ireland/Documents/Proposals/2014/05 HSR Corona/Step 2/'


def norm_data(data):
    # Normalize
    data_mean = np.mean(data)
    return (data - data_mean) / data_mean


def norm_hanning(data):
    data = norm_data(data)
    # length of the time series
    nt = len(data)

    # Multiply by the Hanning window
    return data * np.hanning(nt)


def equally_spaced(nt, duration):
    average_cadence = duration / np.float64(nt - 1)
    return average_cadence * np.arange(0, nt)


#
# Create a data structure for the theoretical data
#
def theory_TS(subsample):

    location = '/home/ireland/ts/sav/NV/Jack_model_TS.sav'

    nvdata = readsav(location)

    output = {}
    for i, k in enumerate(nvdata.keys()):
        # get the data and subsample it
        data = nvdata[k][::subsample]

        # Normalize
        data = norm_data(data)

        # time range
        t = subsample * np.arange(0, len(data))

        wave = k[1:]
        output[wave] = {"wb": wave + '$\AA$',
                     "t": t,
                     "data": data}

    return output


#
# Create a data structure for the TS data
#
def data_TS():

    location = '/home/ireland/ts/sav/NV/Jack_data_TS.sav'
    indata = readsav(location)
    output = {}
    for i, k in enumerate(indata.keys()):
        # get the data and extract the samples and the sample times
        nvdata = indata[k][0, :]
        nvt = np.float64(indata[k][1, :])
        nvt = nvt - nvt[0]
        nt = len(nvdata)

        # resample the data to the average cadence
        t = equally_spaced(nt, nvt[-1])
        data = np.interp(t, nvt, nvdata)

        # Normalize and multiply by the Hanning window
        data = norm_data(data)

        wave = k[4:]
        output[wave] = {"wb": wave + '$\AA$',
                     "t": t,
                     "data": data}

    return output


#
# Create a data structure for the TS2 data
#
def data_TS2():
    location = '/home/ireland/ts/sav/NV/Jack_data_TS2.sav'
    nvdata = readsav(location)

    # Identify the data keys
    k = nvdata.keys()
    datakey = []
    utkey = []
    wavebandkey = []
    for kk in k:
        if 'ut' in kk:
            utkey.append(kk)

        if 'data' in kk:
            datakey.append(kk)

    for kk in datakey:
        wavebandkey.append(kk[4:])

    tspair = {}
    for kk in wavebandkey:
        tspair[kk] = {"time": 'ut' + kk,
                      "data": 'data' + kk}

    output = {}
    for i, kk in enumerate(tspair.keys()):
        # get the data and time series
        data = np.float64(nvdata[tspair[kk]["data"]])
        nvt = np.float64(nvdata[tspair[kk]["time"]] * 60 * 60)
        nvt = nvt - nvt[0]

        # resample to get evenly spaced data at 1800 times
        t = equally_spaced(len(nvt), nvt[-1])
        data = np.interp(t, nvt, data)

        # Normalize
        data = norm_data(data)

        output[kk] = {"data": data, "t": t, 'wb': kk + '$\AA$'}

    return output
