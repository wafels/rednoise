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

# A single event - the integral under the curve is equal to 'A'
def event(t, A, T):
    return A * np.exp(-t / T) / T


# number of events
nevent = 1000