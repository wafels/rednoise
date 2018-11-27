import os
from copy import deepcopy
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

resample_name = 'vk'


class SampleTimes:
    def __len__(self):
        return len(self.sample_times)

    def __init__(self, initial_time, sample_times):
        self.initial_time = initial_time
        self.sample_times = sample_times
        self.duration = sample_times[-1] - sample_times[0]
        self.cadences = self.sample_times[1:] - self.sample_times[0:-1]
        self.mean_cadence = self.duration / (len(self) - 1)
        self.evenly_sampled = np.linspace(self.sample_times[0].value,
                                          self.sample_times[-1].value,
                                          len(self)) * u.s

# Find the latest initial time
initial_times = dict()
latest_initial_time = Time("1970-01-01")
for key in sample_times_keys:
    values = sample_times_indices[key]
    initial_time = Time(values[0][5])
    initial_times[key] = initial_time
    if initial_time > latest_initial_time:
        latest_initial_time = initial_time

# Find the earliest final time
earliest_final_time = Time.now()
for key in sample_times_keys:
    values = sample_times_indices[key]
    initial_time = Time(values[-1][5])
    if initial_time < earliest_final_time:
        earliest_final_time = initial_time

save_directory = "{:s}/{:s}".format(directory, resample_name)

if not os.path.isdir(save_directory):
    os.makedirs(save_directory)
save_filename = resample_name + '.coobserved_time_range'
save_filepath = os.path.join(save_directory, save_filename)
f = open(save_filepath, "w")
f.write(str(latest_initial_time))
f.write(str(earliest_final_time))
f.close()
print('Co-observed time range is {:s} to {:s}.'.format(str(latest_initial_time),
                                                       str(earliest_final_time)))
save_filename = resample_name + '.initial_times'
save_filepath = os.path.join(save_directory, save_filename)
f = open(save_filepath, "w")
for key in initial_times.keys():
    s = "{:s}, {:s}\n".format(key, str(initial_times[key]))
    f.write(s)
f.close()

# Calculate the sample times and resample the data
st = dict()
for key in sample_times_keys:
    values = sample_times_indices[key]
    sample_times = [(Time(v[5]) - latest_initial_time).to(u.s).value for v in values] * u.s
    st[key] = SampleTimes(latest_initial_time, sample_times)
    data_filepath = os.path.join(directory, data_filenames[key])
    print('Reading and resampling {:s}'.format(data_filepath))
    dfp = readsav(data_filepath)
    data_key = list(dfp.keys())[0]
    data = dfp[data_key]
    data_shape = data.shape
    this_st = st[key].sample_times.value
    this_even_st = st[key].evenly_sampled.value

    # Replace the data with data that has been interpolated
    # onto an evenly sampled cadence.
    resampled_data = np.zeros_like(data)
    for i in range(0, data_shape[1]):
        for j in range(0, data_shape[2]):
            f = interpolate.interp1d(this_st, data[:, i, j])
            resampled_data[:, i, j] = f(this_even_st)[:]
    # Save the interpolated data
    save_filename = "{:s}.{:s}".format(data_filenames[key], resample_name)
    save_filepath = os.path.join(save_directory, save_filename)
    np.savez(save_filepath, data=resampled_data, t=this_even_st)
    save_filename = "{:s}.{:s}".format(data_filenames[key], 'initial_time')
    save_filepath = os.path.join(save_directory, save_filename)
    f = open(save_filepath, 'w')
    f.write(str(initial_times[key]))
    f.close()
