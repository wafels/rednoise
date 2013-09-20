"""
Do a wavelet analysis of purely red-noisy time-series data.
"""

import numpy as np
from rnsimulation import TimeSeries, SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
import wav as wavelet
import pylab

# interactive mode
plt.ion()

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Create some fake data
# -----------------------------------------------------------------------------

red_noise = False

dt = 12.0
nt = 300
np.random.seed(seed=1)
model_param = [10.0, 1.77, -100.0]
pls1 = SimplePowerLawSpectrumWithConstantBackground(model_param,
                                                    nt=nt,
                                                    dt=dt)
data = TimeSeriesFromPowerSpectrum(pls1).sample
t = dt * np.arange(0, nt)
amplitude = 0.0
data = data + amplitude * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)

# Create a time series object
ts = TimeSeries(t, data)
ts.label = 'emission'
ts.units = 'arb. units'
ts.name = 'simulated data [n=%4.2f]' % (model_param[1])

# Get the normalized power and the positive frequencies
iobs = ts.PowerSpectrum.Npower
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Wavelet transform using a white noise background
# -----------------------------------------------------------------------------
var = ts.data
# Range of periods to average
avg1, avg2 = (150.0, 400.0)

# Significance level
slevel = 0.95

# Standard deviation
std = var.std()

# Variance
std2 = std ** 2

# Calculating anomaly and normalizing
var = (var - var.mean()) / std

# Number of measurements
N = var.size

# Time array in seconds
time = ts.SampleTimes.time + 0.001

# Four sub-octaves per octaves
dj = 0.25

#2 * dt # Starting scale, here 6 months
s0 = -1

#7 / dj # Seven powers of two with dj sub-octaves
J = -1

#alpha = 0.0 # Lag-1 autocorrelation for white noise

alpha, _, _ = wavelet.ar1(var)
noise_type = 'red noise'

# Morlet mother wavelet with wavenumber=6
mother = wavelet.Morlet(6.)
#mother = wavelet.Mexican_hat() # Mexican hat wavelet, or DOG with m=2
#mother = wavelet.Paul(4) # Paul wavelet with order m=4
#mother = wavelet.DOG(6) # Derivative of the Gaussian, with m=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother)

# Normalized wavelet power spectrum
power = (abs(wave)) ** 2

# FFT power spectrum
fft_power = std2 * abs(fft) ** 2
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]

# Where ratio > 1, power is significant
sig95 = power / sig95

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2 * power.mean(axis=1)

# Correction for padding at edges
dof = N - scales
glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                       significance_level=slevel, dof=dof, wavelet=mother)

# Scale average between avg1 and avg2 periods and significance level
sel = pylab.find((period >= avg1) & (period < avg2))
Cdelta = mother.cdelta
scale_avg = (scales * np.ones((N, 1))).transpose()
# As in Torrence and Compo (1998) equation 24
scale_avg = power / scale_avg
scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
                            significance_level=slevel, dof=[scales[sel[0]],
                            scales[sel[-1]]], wavelet=mother)

# The following routines plot the results in four different subplots containing
# the original series anomaly, the wavelet power spectrum, the global wavelet
# and Fourier spectra and finally the range averaged wavelet spectrum. In all
# sub-plots the significance levels are either included as dotted lines or as
# filled contour lines.
pylab.close('all')
pylab.ion()
fontsize = 'medium'
params = {'text.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'text.usetex': True
         }

# Plot parameters
pylab.rcParams.update(params)
figprops = dict(figsize=(11, 8), dpi=72)
fig = pylab.figure(**figprops)

# ---------------------------------------------------------------------------
# First sub-plot, the original time series anomaly.
ax = pylab.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, var, 'k', linewidth=1.5)
ax.set_title('a) %s' % (ts.name, ))
if ts.units != '':
    ax.set_ylabel(r'%s [$%s$]' % (ts.label, ts.units,))
else:
    ax.set_ylabel(r'%s' % (ts.label, ))
# ---------------------------------------------------------------------------
# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
bx = pylab.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')
bx.contour(time, np.log2(period), sig95, [-99, 1], colors='k',
           linewidths=2., label='95%')
bx.axhline(np.log2(300.0), label='5 mins', linewidth=2, color='w', linestyle='--')
bx.axhline(np.log2(180.0), label='3 mins', linewidth=2, color='w')
legend = bx.legend(shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.8')

coi[-1] = 0.001
bx.fill(np.concatenate([time[:1] - dt, time, time[-1:] + dt, time[-1:] + dt, time[:1] - dt, time[:1] - dt]),
        np.log2(np.concatenate([[1e-9], coi, [1e-9], period[-1:], period[-1:], [1e-9]])),
        'k',
        alpha=0.3,
        hatch='x')
bx.set_title('b) Wavelet Power Spectrum (%s) assuming %s' % (mother.name, noise_type))
bx.set_ylabel('Period (seconds)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks)
bx.invert_yaxis()

# Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
# avg1 and avg2 parameters

Bx = pylab.axes([0.1, 0.07, 0.65, 0.22], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
Bx.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')
Bx.contour(time, np.log2(period), sig95, [-99, 1], colors='k',
           linewidths=2., label='95%')
Bx.axhline(np.log2(300.0), label='5 mins', linewidth=2, color='w', linestyle='--')
Bx.axhline(np.log2(180.0), label='3 mins', linewidth=2, color='w')
legend = Bx.legend(shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.8')

coi[-1] = 0.001
Bx.fill(np.concatenate([time[:1] - dt, time, time[-1:] + dt, time[-1:] + dt, time[:1] - dt, time[:1] - dt]),
        np.log2(np.concatenate([[1e-9], coi, [1e-9], period[-1:], period[-1:], [1e-9]])),
        'k',
        alpha=0.3,
        hatch='x')
Bx.set_title('b) Wavelet Power Spectrum (%s) assuming %s' % (mother.name, noise_type))
Bx.set_ylabel('Period (seconds)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
Bx.set_yticks(np.log2(Yticks))
Bx.set_yticklabels(Yticks)
Bx.invert_yaxis()


# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
cx = pylab.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, np.log2(period), 'k--')
cx.plot(fft_power, np.log2(1. / fftfreqs), '-', color=[0.7, 0.7, 0.7],
        linewidth=1.)
cx.plot(glbl_power, np.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
if ts.units != '':
    cx.set_xlabel(r'Power [$%s^2$]' % (ts.units, ))
else:
    cx.set_xlabel(r'Power')
#cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels(Yticks)
pylab.setp(cx.get_yticklabels(), visible=False)
cx.invert_yaxis()


"""

dx = pylab.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(time, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) $%d$-$%d$ second scale-averaged power' % (avg1, avg2))
dx.set_xlabel('Time (seconds)')
if ts.units != '':
    dx.set_ylabel(r'Average variance [$%s$]' % (ts.units, ))
else:
    dx.set_ylabel(r'Average variance')
#
ax.set_xlim([time.min(), time.max()])
"""
#
pylab.draw()
pylab.show()

