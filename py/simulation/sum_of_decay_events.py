#
# Simulation of the effect of a large number of decaying events
#
# Based on "Self-organized criticality in astrophysics", Aschwanden 2011
# chapter 4
#
import numpy as np


def spectrum_per_event(nu, timescale):
    return 1.0 / (1.0 + (2*np.pi*nu*timescale)**2)


def event_energy(gamma, timescale):
    return timescale**(1.0 + gamma)


def size_distribution(alpha, energy):
    return energy**-alpha


def timescale_distribution(t0, t1):
    return np.random.uniform(low=t0, high=t1)

#
# Create a power spectrum from a large number of component events
#
def spectrum(t0, t1, gamma, alpha, nu, nsamples=100):
    """
    :param t0: shortest timescale of the events
    :param t1: longest timescale of the events
    :param gamma: event energy as a function of event time scale
    :param alpha: number of events of a given energy
    :param nu: frequency range we want to calculate the
    :param nsamples: number of samples from the time-scale distribution
    :return: the theoretical power spectrum summed over these events
    """
    # Storage for the power spectrum
    power_spectrum = np.zeros_like(nu)

    # Sum over the timescales
    i = 0
    while i < nsamples:
        i += 1
        timescale = timescale_distribution(t0, t1)
        energy_per_event = event_energy(gamma, timescale)
        number_of_events = size_distribution(alpha, energy_per_event)
        power_spectrum += number_of_events * energy_per_event * spectrum_per_event(nu, timescale)

    return power_spectrum

class Spectrum:
    def __init__(self, gamma, alpha, t0, t1):
        self.gamma = gamma
        self.alpha = alpha
        self.t0 = t0
        self.t1 = t1

    def spectrum_per_event(self, nu, timescale):
        return 1.0 / (1.0 + (2*np.pi*nu*timescale)**2)

    def event_energy(self, timescale):
        return timescale**(1.0 + self.gamma)

    def size_distribution(self, energy):
        return energy**-self.alpha

    def timescale_distribution(self):
        return np.random.uniform(low=self.t0, high=self.t1)

    def calculate(self, nu, nsamples):
        # Storage for the power spectrum
        power_spectrum = np.zeros_like(nu)

        # Sum over the timescales
        i = 0
        while i < nsamples:
            i += 1
            timescale = self.timescale_distribution()
            energy_per_event = self.event_energy(self.gamma, timescale)
            number_of_events = self.size_distribution(self.alpha, energy_per_event)
            power_spectrum += number_of_events * energy_per_event * self.spectrum_per_event(nu, timescale)

        return power_spectrum