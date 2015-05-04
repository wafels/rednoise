#
# Simulation of the effect of a large number of decaying events
#
# Based on "Self-organized criticality in astrophysics", Aschwanden 2011
# chapter 4
#
import numpy as np


class SummedEventPowerSpectrum:
    def __init__(self, gamma, alpha, t0, t1):
        # Dependence of the energy of an event on its timescale
        self.gamma = gamma
        # Dependence of the number of events as a function of energy
        self.alpha = alpha
        # Minimum and maximum timescales of the event
        self.t0 = t0
        self.t1 = t1

    def spectrum_per_event(self, nu, timescale):
        return 1.0 / (1.0 + (2*np.pi*nu*timescale)**2)

    # Energy of each event as a function of its characteristic timescale
    def event_energy(self, timescale):
        return timescale**(1.0 + self.gamma)

    # Number of events as a function of their energy
    def number_distribution(self, energy):
        return energy**-self.alpha

    # Distribution of timescales of events
    def timescale_distribution(self):
        return np.random.uniform(low=self.t0, high=self.t1)

    # Calculate the resultant power spectrum in the frequency range nu, using
    # 'nsamples' from the time scale distribution.
    def calculate(self, nu, nsamples):
        # Storage for the power spectrum
        power_spectrum = np.zeros_like(nu)

        # Sum over the timescales
        i = 0
        while i < nsamples:
            i += 1
            timescale = self.timescale_distribution()
            energy_per_event = self.event_energy(self.gamma, timescale)
            number_of_events = self.number_distribution(self.alpha, energy_per_event)
            power_spectrum += number_of_events * energy_per_event * self.spectrum_per_event(nu, timescale)

        return power_spectrum