import os
import numpy as np
from scipy.io import readsav
from scipy import interpolate
from astropy.time import Time
import astropy.units as u

directory = os.path.expanduser('~/Data/ts/viall_klimchuk')

sample_times_filename = 'jun19_index_paul.sav'

sample_times_filepath = os.path.join(directory, sample_times_filename)

sample_times_indices = readsav(sample_times_filepath)

sample_times_keys = list(sample_times_indices.keys())

data_filenames = {'index171': 'jun19_171_derot_paul.sav',
                  'index131': 'jun19_131_derot_paul.sav',
                  'index193': 'jun19_193_derot_paul.sav',
                  'index211': 'jun19_211_derot_paul.sav',
                  'index335': 'jun19_335_derot_paul.sav',
                  'index094': 'jun19_94_derot_paul.sav'}


class SampleTimes:
    def __len__(self):
        return len(self.sample_times)

    def __init__(self, initial_time, sample_times):
        self.initial_time = initial_time
        self.sample_times = sample_times
        self.duration = sample_times[-1] - sample_times[0]
        self.cadences = self.sample_times[1:] - self.sample_times[0:-1]
        self.mean_cadence = self.duration / (len(self) - 1)
        self.evenly_sampled = np.linspace(0, self.duration.value, len(self)) * u.s

# Find the earliest initial time
first_initial_time = Time.now()
for key in sample_times_keys:
    values = sample_times_indices[key]
    initial_time = Time(values[0][5])
    print(initial_time)
    if initial_time < first_initial_time:
        first_initial_time = initial_time

# Calculate the sample times and resample the data
st = dict()
for key in sample_times_keys:
    values = sample_times_indices[key]
    sample_times = [(Time(v[5]) - first_initial_time).to(u.s).value for v in values] * u.s
    st[key] = SampleTimes(first_initial_time, sample_times)
    data_filepath = os.path.join(directory, data_filenames[key])
    dfp = readsav(data_filepath)
    data_key = list(dfp.keys())[0]
    data = dfp[data_key]
    data_shape = data.shape
    this_st = st[key].sample_times.value
    this_even_st = st[key].evenly_sampled.value

    # Replace the data with data that has been interpolated
    # onto an evenly sampled cadence.
    for i in range(0, data_shape[1]):
        for j in range(0, data_shape[2]):
            f = interpolate.interp1d(this_st, data[:, i, j])
            data[:, i, j] = f(this_even_st)[:]

    # Save the interpolated data
    np.save()