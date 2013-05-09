"""
Testing RN
"""

import numpy as np
import matplotlib.pyplot as plt
import rn_utils

# Get some power law noise.  Make sure it is
# highly oversampled
nsample = 10000
desired_length = 300
power_law_index = 0.0
data = rn_utils.power_law_noise1(nsample, 1, power_law_index)
test_data = data[nsample/2-desired_length/2:nsample/2+desired_length/2]
test_data = test_data - np.mean(test_data)

# plot the data itself
plt.figure(1)
plt.plot(test_data)
plt.xlabel('sample number')
plt.ylabel('emission (constant background subtracted)')
plt.title('Simulated data with p=' + str(power_law_index))
plt.show()

# Power spectrum
input_power_spectrum = (np.absolute(np.fft.fft(test_data))) ** 2

# Analysis frequencies and power
n = len(test_data)
dt = 12
fftfreq = np.fft.fftfreq(n, dt)
analysis_frequencies = fftfreq[fftfreq >= 0][1:-1]
analysis_power = input_power_spectrum[fftfreq >= 0][1:-1]


# get a quick estimate assuming the data is a power law only.
# the [0] entry is the power law index, the [1] entry is the
# value at np.log(observed_power_spectrum) = 0.0
coefficients = np.polyfit(np.log(analysis_frequencies),
                           np.log(analysis_power),
                           1)
m_estimate = -coefficients[0]
c_estimate = np.exp(coefficients[1])

# Fit to the power spectrum
power_fit = c_estimate * analysis_frequencies ** (-m_estimate)

# plot the power spectrum and the quick fit
plt.figure(2)
plt.loglog(analysis_frequencies, analysis_power, label='observed power')
plt.loglog(analysis_frequencies, power_fit, label='fit: ' + str(m_estimate))
plt.xlabel('frequency')
plt.ylabel('power')
plt.title('Simulated data with power law index = ' + str(power_law_index))
plt.legend()
plt.show()
