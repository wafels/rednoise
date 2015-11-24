#
# Test the broken power law fit
#
import numpy as np
import rnspectralmodels2
import matplotlib.pyplot as plt

plt.ion()
nf = 900

df = 1.0 / (12 * 1800.0)

a = [np.log(1.0), 6.0, 20.0, 1.5, np.log(0.000001)]

f = 1.0 * np.arange(1, nf)

bpl = rnspectralmodels2.BrokenPowerLawPlusConstant()

true_power = bpl.power(a, f)

noisy_power = np.random.chisquare(2, nf-1) * true_power

guesstimate = bpl.guess(f, noisy_power)

guess_power = bpl.power(guesstimate, f)

plt.loglog(f, noisy_power, label='noisy power')
plt.loglog(f, true_power, label='true power')
plt.loglog(f, guess_power, label='guess power')
plt.legend()
plt.show()


