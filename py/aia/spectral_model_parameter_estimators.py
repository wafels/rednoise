import numpy as np


# Assume a power law in power spectrum - Use a Bayesian marginal distribution
# to calculate the probability that the power spectrum has a power law index
# 'n'.
def bayeslogprob(f, I, n, m):
    """
    Return the log of the marginalized Bayesian posterior of an observed
    Fourier power spectra fit with a model spectrum Af^{-n}, where A is a
    normalization constant, f is a normalized frequency, and n is the power law
    index at which the probability is calculated. The marginal probability is
    calculated using a prior p(A) ~ A^{m}.
    The function returns log[p(n)] give.  The most likely value of 'n' is the
    maximum value of p(n).
    f : normalized frequencies
    I : Fourier power spectrum
    n : power law index of the power spectrum
    m : power law index of the prior p(A) ~ A^{m}
    """
    N = len(f)
    term1 = n * np.sum(np.log(f))
    term2 = (N - m - 1) * np.log(np.sum(I * f ** n))
    return term1 - term2


# Find the most likely power law index given a prior on the amplitude of the
# power spectrum.
def most_probable_power_law_index(f, I, m, n):
    blp = np.zeros_like(n)
    for inn, nn in enumerate(n):
        blp[inn] = bayeslogprob(f, I, nn, m)
    return n[np.argmax(blp)]


class InitialParameterEstimatePlC(object):
    def __init__(self, f, p, ir=None, ar=None, br=None,
                 bayes_search=(0, np.linspace(0, 20, 100))):
        """
        Estimate of three parameters of the power law + constant observation model - the amplitude,
        the power law index, and the background value.

        Parameters
        ----------
        f :
            Positive frequencies of the power law spectrum

        p :
            Power at the frequencies

        ir :


        ar :


        br :


        bayes_search : ~tuple


        """
        self.f = f
        self.p = p
        self.ir = ir
        self.ar = ar
        self.br = br
        self.bayes_search = bayes_search

        # The bit in-between the low and high end of the spectrum to estimate the power law index.
        self._index = most_probable_power_law_index(self.f[self.ir[0]:self.ir[1]]/self.f[ir[0]],
                                                    self.p[self.ir[0]:self.ir[1]],
                                                    bayes_search[0], bayes_search[1])

        # Use the low-frequency end to estimate the amplitude, normalizing for the first frequency
        observed_power = self.p[self.ar[0]:self.ar[1]]
        observed_freqs = self.f[self.ar[0]:self.ar[1]] ** self._index
        self._amplitude = np.exp(np.mean(np.log(observed_power*observed_freqs)))

        # Use the high frequency part of the spectrum to estimate the constant value.
        self._background = np.exp(np.mean(np.log(self.p[self.br[0]:self.br[1]])))

    @property
    def index(self):
        return self._index

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def background(self):
        return self._background


class OldSchoolInitialParameterEstimatePlC(object):
    def __init__(self, f, p, ir=(0, 50), ar=(0, 5), br=(-50, -1),
                 bayes_search=(0.0, np.arange(0, 4, 0.01))):
        """
        Estimate of three parameters of the power law + constant observation model - the amplitude,
        the power law index, and the background value.

        Parameters
        ----------
        f :
            Positive frequencies of the power law spectrum

        p :
            Power at the frequencies

        ir :


        ar :


        br :


        bayes_search : ~tuple


        """
        self.f = f
        self.p = p
        self.ir = ir
        self.ar = ar
        self.br = br
        self.bayes_search = bayes_search

        # The bit in-between the low and high end of the spectrum to estimate the power law index.
        self._index = most_probable_power_law_index(self.f[self.ir[0]:self.ir[1]],
                                                    self.p[self.ir[0]:self.ir[1]],
                                                    bayes_search[0], bayes_search[1])

        # Use the low-frequency end to estimate the amplitude, normalizing for the first frequency
        self._amplitude = np.mean(self.p[self.ar[0]:self.ar[1]])

        # Use the high frequency part of the spectrum to estimate the constant value.
        self._background = np.exp(np.mean(np.log(self.p[self.br[0]:self.br[1]])))

    @property
    def index(self):
        return self._index

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def background(self):
        return self._background
