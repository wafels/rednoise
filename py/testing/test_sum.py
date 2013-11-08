"""
Test adding up a bunch of randomly sized decays
Event times are Poisson distributed
Event sizes follow the Aschwanden model
Event number distribution follow a power law

Note that in scipy and numpy the power law model we want is called the
Pareto distribution.

"""
 
# Test 1
import rnspectralmodels
import rnsimulation
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from timeseries import TimeSeries

# A single event - the integral under the curve is equal to 'A'
def event(t, t0, A, T):
    y = np.zeros_like(t)
    y[t > t0] = A * np.exp(-(t[t > t0] - t0) / T) / T
    return y


# number of time-scales to sample
nscale = 10000

# N(E) - number of events and their energy N(E) ~ E^{-alpha_e}
alpha_e = 1.0

# E(T) ~ T^{1+gamma}
gamma = 1.0

# power
power = alpha_e * (1.0 + gamma) - gamma

# Number of events at each time scale
number = 10 * np.random.pareto(power, size=nscale)

# Nearest integer number of events
number_int = np.rint(number)

# Form the event list
event_list = np.array([])
for n in number_int:
    if n >= 1:
        tmp = n * np.ones(n)
        event_list = np.concatenate((event_list, tmp)).flatten()

# Event list
np.random.shuffle(event_list)

# Poisson mean time between events
lam = 1.0

#
t = 0.01 * np.arange(lam * event_list.size)
emission = np.zeros_like(t)
start_time = 0.0
for ev in event_list:
    T = (1.0 * ev) ** (-power)
    energy = T ** (1.0 + gamma)
    start_time = start_time + 0.01 * np.random.poisson(lam=lam)
    emnew = event(t, start_time, energy, T)
    if np.max(emnew) >= 0.0:
        emission = emission + emnew
    else:
        print('Not big enough')

ts = TimeSeries(t[::10], emission[::10])
ts.peek_ps()
plt.loglog()
plt.show()
