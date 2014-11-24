"""
Do a wavelet analysis of purely red-noisy time-series data.
"""

import numpy as np
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import pyplot as plt
from timeseries import TimeSeries
import wav as wavelet
import pylab
import matplotlib.cm as cm
import fakedata, fitspecfig5
from scipy.stats import chi2

# interactive mode
plt.ion()

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Create some fake data
# -----------------------------------------------------------------------------


t, data, model_param = fakedata.fakedata()
nt = t.size
dt = t[1] - t[0]
M0 = fitspecfig5.do(t, data, window=False)
meanfit = M0.stats()['fourier_power_spectrum']['mean']

"""
# Running average filter
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


mvav = moving_average(data, n=84)
data = data - mvav

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w = np.ones(window_len, 'd')
    else:
            w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]

tsoriginal = TimeSeries(t, data)
plt.figure(10)
tsoriginal.peek()
"""

meandata = np.mean(data)

# relative
data = (data - meandata) / meandata


#data = data - smooth(data, window_len=84)

# Create a time series object
ts = TimeSeries(t, data)
ts.label = 'emission'
ts.units = 'arb. units'
ts.name = 'simulated data [n=%4.2f]' % (model_param[1])

# Get the normalized power and the positive frequencies
iobs = ts.PowerSpectrum.ppower
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Wavelet transform using a white noise background
# -----------------------------------------------------------------------------
var = ts.data
# Range of periods to average
avg1, avg2 = (150.0, 400.0)

# Significance level
slev = 95.0
slevel = slev / 100.0

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


# -------------------------------------------------------------------------
# White noise
signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, 0.0,
                        significance_level=slevel, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]

# Where ratio > 1, power is significant
sig95 = power / sig95

# Calculates the global wavelet spectrum and determines its significance level.
glbl_power = std2 * power.mean(axis=1)

# Correction for padding at edges
dof = N - scales
glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, 0.0,
                       significance_level=slevel, dof=dof, wavelet=mother)

max_glbl_power = np.max(np.abs(glbl_power))
glbl_power = glbl_power / max_glbl_power

# -------------------------------------------------------------------------
# Red noise
rsignif, rfft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
rsig95 = np.ones([1, N]) * rsignif[:, None]
rsig95 = np.ones([1, N]) * meanfit[:, None] * chi2.ppf(slevel, 2) / 2.0

# Where ratio > 1, power is significant
rsig95 = power / rsig95

# Calculates the global wavelet spectrum and determines its significance level.
rglbl_power = std2 * power.mean(axis=1)

# Correction for padding at edges
dof = N - scales
rglbl_signif, rtmp = wavelet.significance(std2, dt, scales, 1, alpha,
                       significance_level=slevel, dof=dof, wavelet=mother)


rglbl_power = rglbl_power / max_glbl_power


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

fig_width = 0.65
fig_height = 0.23

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
ax = pylab.axes([0.1, 0.70, fig_width, fig_height])
#ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, var, 'k', linewidth=1.5)
ax.set_title('(a) %s' % (ts.name, ))
if ts.units != '':
    ax.set_ylabel(r'%s [$%s$]' % (ts.label, ts.units,))
else:
    ax.set_ylabel(r'%s' % (ts.label, ))
# ---------------------------------------------------------------------------
# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
bx = pylab.axes([0.1, 0.38, fig_width, fig_height], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
YYY = np.log2(period)
bx.contourf(time, YYY, np.log2(power), np.log2(levels),
            extend='both', cmap=cm.coolwarm)
bx.contour(time, np.log2(period), sig95, [-99, 1], colors='k',
           linewidths=2., label='95%')
bx.axhline(np.log2(300.0), label='5 mins', linewidth=2, color='k', linestyle='--')
bx.axhline(np.log2(180.0), label='3 mins', linewidth=2, color='k')
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
conf_level_string = '95' + r'$\%$'
bx.set_title('(b) Wavelet power spectrum compared to white noise [' + conf_level_string + ' conf. level]')
bx.set_ylabel('Period (seconds)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks)
bx.invert_yaxis()


# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
cx = pylab.axes([0.77, 0.38, 0.2, fig_height], sharey=bx)
cx.plot(glbl_signif / max_glbl_power, np.log2(period), 'k--', label='white noise')
#cx.plot(fft_power, np.log2(1. / fftfreqs), '-', color=[0.7, 0.7, 0.7],
#        linewidth=1.)
cx.plot(glbl_power, np.log2(period), 'k-', linewidth=1.5, label='power')
cx.set_title('(c) Global wavelet spectrum')
if ts.units != '':
    cx.set_xlabel(r'Power [arb units.]')
else:
    cx.set_xlabel(r'Power')
#cx.set_xlim([0, glbl_power.max() + std2])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels(Yticks)
pylab.setp(cx.get_yticklabels(), visible=False)
cx.invert_yaxis()
clegend = cx.legend(shadow=True, fontsize=10)
cframe = clegend.get_frame()


# RED Fourth sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.

Bx = pylab.axes([0.1, 0.07, fig_width, fig_height], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
Bx.contourf(time, YYY, np.log2(power), np.log2(levels), extend='both', cmap=cm.coolwarm)
Bx.contour(time, np.log2(period), rsig95, [-99, 1], colors='k',
           linewidths=2., label='95%')
Bx.axhline(np.log2(300.0), label='5 mins', linewidth=2, color='k', linestyle='--')
Bx.axhline(np.log2(180.0), label='3 mins', linewidth=2, color='k')
legend = Bx.legend(shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.8')

coi[ coi< 2**YYY.min() ] = 2**YYY.min()#0.001
lim = coi.min()#[-1]#1e-9
Bx.fill(np.concatenate([time[:1] - dt, time, time[-1:] + dt, time[-1:] + dt, time[:1] - dt, time[:1] - dt]),
        np.log2(np.concatenate([[lim], coi, [lim], period[-1:], period[-1:], [lim]])),
        'k',
        alpha=0.3,
        hatch='x')
#pc = r'\%'
Bx.set_title('(d) Wavelet power spectrum compared to power-law power-spectrum noise [' + conf_level_string + ' conf. level]')
Bx.set_ylabel('Period (seconds)')
Bx.set_xlabel('Time (seconds) [%i samples]' % (nt))
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                          np.ceil(np.log2(period.max())))
Bx.set_yticks(np.log2(Yticks))
Bx.set_yticklabels(Yticks)
Bx.invert_yaxis()


# Fifth sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra.
Cx = pylab.axes([0.77, 0.07, 0.2, fig_height], sharey=bx)
Cx.plot(rglbl_signif/max_glbl_power, np.log2(period), 'k--', label='power-law power-spectrum')
#Cx.plot(fft_power, np.log2(1. / fftfreqs), '-', color=[0.7, 0.7, 0.7],
#        linewidth=1.)
Cx.plot(rglbl_power, np.log2(period), 'k-', linewidth=1.5, label='power')
Cx.set_title('(e) Global wavelet spectrum')
if ts.units != '':
    Cx.set_xlabel(r'Power [arb. units]')
else:
    Cx.set_xlabel(r'Power')
#Cx.set_xlim([0, glbl_power.max() + std2])
Cx.set_ylim(np.log2([period.min(), period.max()]))
Cx.set_yticks(np.log2(Yticks))
Cx.set_yticklabels(Yticks)
pylab.setp(Cx.get_yticklabels(), visible=False)
Cx.invert_yaxis()
Clegend = Cx.legend(shadow=True, fontsize=10)
Cframe = Clegend.get_frame()





ax.set_xlim([time.min(), time.max()])

#
pylab.draw()
pylab.show()

