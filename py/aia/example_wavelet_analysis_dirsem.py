"""
Do a wavelet analysis of purely red-noisy time-series data. 
"""
import os
import numpy as np
from rnsimulation import SimplePowerLawSpectrumWithConstantBackground, TimeSeriesFromPowerSpectrum
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
import matplotlib

from timeseries import TimeSeries
import wav as wavelet
import pylab
import matplotlib.cm as cm

from astropy.convolution import convolve as ap_convolve
from astropy.convolution import Box1DKernel

outdir = os.path.expanduser('~/Documents/Talks/2014/DirSem/')

wvt_use_cm = cm.seismic
wvt_linecolor = 'w'
wvt_linewidth = 4
wvt_coi_color = 'k'
wvt_alpha = 0.5
wvt_hatch = 'x'
mvscale = 500.0
slev = 95.0

fake = False

# interactive mode
plt.ion()

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Create some fake data
# -----------------------------------------------------------------------------

if fake:
    obs = 'fake'
    dt = 12.0
    nt = 300
    np.random.seed(seed=1)
    model_param = [10.0, 1.77, -100.0]
    pls1 = SimplePowerLawSpectrumWithConstantBackground(model_param,
                                                        nt=nt,
                                                        dt=dt)
    data = np.abs(TimeSeriesFromPowerSpectrum(pls1).sample)
    t = dt * np.arange(0, nt)
    amplitude = 0.0
    data = data + amplitude * (data.max() - data.min()) * np.sin(2 * np.pi * t / 300.0)

    data = data / np.std(data)

    # Linear increase
    lin = 0.0
    data = data + lin * t / np.max(t)

    # constant
    constant = 10.0
    data = data + constant
else:
    obs = 'sunspot171.npy'
    filename = os.path.join(outdir, obs)
    data = np.load(filename)
    nt = data.size
    dt = 12.0
    t = dt * np.arange(0, nt)
    model_param = [10.0, 1.77, -100.0]

# Running average filter
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

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

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#
# Background subtracted
#
mvav = ap_convolve(data, Box1DKernel(np.rint(mvscale/dt)))
subdata = data - mvav
subdata = subdata - np.mean(subdata)
tssubdata = TimeSeries(t, subdata)


#tsoriginal = TimeSeries(t, data)

#
# Relative data
#
# mean of the data
meandata = np.mean(data)
# relative
data = (data - meandata) / meandata
# Create a time series object
ts = TimeSeries(t, data)
ts.label = 'emission'
ts.units = 'arb. units'
ts.name = obs

# Get the normalized power and the positive frequencies
iobs = ts.PowerSpectrum.ppower
this = ([ts.PowerSpectrum.frequencies.positive, iobs],)

# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# Wavelet transform using a white noise background
# -----------------------------------------------------------------------------
# data
var = tssubdata.data
# Time array in seconds
time = ts.SampleTimes.time + 0.001

# Range of periods to average
avg1, avg2 = (150.0, 400.0)

# Significance level
slevel = slev / 100.0

# Standard deviation
std = var.std()

# Variance
std2 = std ** 2

# Calculating anomaly and normalizing
var = (var - var.mean()) / std
# Number of measurements
N = var.size

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

#fig_width = 0.65
#fig_height = 0.23

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
#pylab.rcParams.update(params)
#figprops = dict(figsize=(11, 33), dpi=72)
#fig = pylab.figure(**figprops)

# ---------------------------------------------------------------------------
# First sub-plot, the original time series anomaly.
"""
ax = pylab.axes([0.1, 0.70, fig_width, fig_height])
#ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, var, 'k', linewidth=1.5)
ax.set_title('(a) %s' % (ts.name, ))
if ts.units != '':
    ax.set_ylabel(r'%s [$%s$]' % (ts.label, ts.units,))
else:
    ax.set_ylabel(r'%s' % (ts.label, ))
"""

#
# Simulated data
#
with plt.style.context(("ggplot")):
    matplotlib.rcParams.update({'font.size': 22, 'axes.labelcolor': 'k', 'xtick.color': 'k', 'ytick.color': 'k'})
    plt.figure(1, figsize=(24, 6))
    plt.plot(time, var, linewidth=3.0)
    plt.title('')
    plt.xlabel("time (seconds)")
    plt.ylabel("emission (arb. units)")
    plt.xlim(time[0], time[-1])
    plt.savefig(os.path.join(outdir, obs + '_data.png'))
    plt.close("all")
#
# Power spectrum of simulated data
#
with plt.style.context(("ggplot")):
    tsh = TimeSeries(t, data * np.hanning(nt))
    iobsh = tsh.PowerSpectrum.ppower
    matplotlib.rcParams.update({'font.size': 22, 'axes.labelcolor': 'k', 'xtick.color': 'k', 'ytick.color': 'k'})
    plt.figure(1, figsize=(24, 6))
    plt.loglog(1000 * tsh.PowerSpectrum.frequencies.positive, iobsh, linewidth=3.0, label=obs)
    if fake:
        plt.loglog(1000 * pls1.frequencies, pls1.power() / 10.0 ** 9.4, label=r'$\nu^{-1.77}$', linewidth=3.0)
    plt.axvline(1000.0 / 300.0, label='5 mins.', color='k', linestyle=":", linewidth=3.0)
    plt.axvline(1000.0 / 180.0, label='3 mins.', color='k', linestyle='-', linewidth=3.0)
    plt.title('power spectrum')
    plt.xlabel("frequency " + r"$\nu$" + " (mHz)")
    plt.ylabel("power (arb. units)")
    plt.legend(loc=3)
    plt.savefig(os.path.join(outdir, obs + "_data_ps.png"))
    plt.close("all")
 
#
# Wavelet
#
wv_time = -5 + 0.01 * np.arange(0, 999)
wv_morlet = np.exp(-0.5 * wv_time ** 2) * np.cos(wv_time * 6.0)
with plt.style.context(("ggplot")):
    matplotlib.rcParams.update({'font.size': 22, 'axes.labelcolor': 'k', 'xtick.color': 'k', 'ytick.color': 'k'})
    plt.figure(1, figsize=(10, 6))
    plt.plot(wv_time, wv_morlet, linewidth=3.0)
    plt.title('Morlet wavelet')
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.savefig(os.path.join(outdir, "morlet_wavelet.png"))
    plt.close("all")

# ---------------------------------------------------------------------------
# Second sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.
plt.figure(2, figsize=(12, 6))
matplotlib.rcParams.update({'font.size': 18})
levels = 2.0 ** (-4 + np.arange(0, 8, 0.1))
#levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
YYY = np.log2(period)
CS = plt.contourf(time, YYY, np.log2(sig95), np.log2(levels),
            extend='both', cmap=wvt_use_cm)
CS2 = plt.contour(time, YYY, sig95, [-99, 1], colors='k',
           linewidths=3., label='95%')
plt.axhline(np.log2(300.0), label='5 mins', linewidth=wvt_linewidth, color=wvt_linecolor, linestyle='--')
plt.axhline(np.log2(180.0), label='3 mins', linewidth=wvt_linewidth, color=wvt_linecolor)
legend = plt.legend(frameon=1, shadow=True)
frame = legend.get_frame()
frame.set_color('grey')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\log_{2}($normalized wavelet power$)$')

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
#frame = legend.get_frame()
#frame.set_facecolor('0.8')


coi[coi < period[0]] = period[0]
xfill = np.concatenate([time[:1] - dt, time, time[-1:] + dt, time[-1:] + dt, time[:1] - dt, time[:1] - dt])
yfill = np.concatenate([[period[0]], coi, [period[0]], period[-1:], period[-1:], [period[0]]])
yfill = np.log2(yfill)
plt.fill(xfill, yfill, wvt_coi_color, alpha=wvt_alpha, hatch=wvt_hatch)
conf_level_string = '95' + r'$\%$'
#plt.title('(b) Wavelet power spectrum compared to white noise [' + conf_level_string + ' conf. level]')
plt.ylabel('period (seconds)')
plt.xlabel('time (seconds)')

Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
plt.yticks(np.log2(Yticks), [str(m) for m in Yticks])
#plt.gca().invert_yaxis()
plt.savefig(os.path.join(outdir, obs + ".white_noise_" + str(slev) + ".png"))
plt.tight_layout()
plt.close("all")

"""
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
"""

# RED Fourth sub-plot, the normalized wavelet power spectrum and significance level
# contour lines and cone of influece hatched area.

#Bx = pylab.axes([0.1, 0.07, fig_width, fig_height], sharex=ax)
plt.figure(3, figsize=(12, 6))
matplotlib.rcParams.update({'font.size': 18})
#levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
levels = 2.0 ** (-4 + np.arange(0, 8, 0.1))
CS = plt.contourf(time, YYY, np.log2(rsig95), np.log2(levels), extend='both', cmap=wvt_use_cm)
plt.contour(time, np.log2(period), rsig95, [-99, 1], colors='k',
           linewidths=2., label='95%')
plt.axhline(np.log2(300.0), label='5 mins', linewidth=wvt_linewidth, color=wvt_linecolor, linestyle='--')
plt.axhline(np.log2(180.0), label='3 mins', linewidth=wvt_linewidth, color=wvt_linecolor)
legend = plt.legend(shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
legend = plt.legend(frameon=1, shadow=True)
frame = legend.get_frame()
frame.set_color('grey')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\log_{2}($normalized wavelet power$)$')

coi[coi < 2 ** YYY.min() ] = 2**YYY.min()#0.001
lim = coi.min()#[-1]#1e-9
xfill = np.concatenate([time[:1] - dt, time, time[-1:] + dt, time[-1:] + dt, time[:1] - dt, time[:1] - dt])
yfill = np.log2(np.concatenate([[lim], coi, [lim], period[-1:], period[-1:], [lim]]))
plt.fill(xfill, yfill, wvt_coi_color, alpha=wvt_alpha, hatch=wvt_hatch)
#pc = r'\%'
#plt.title('(d) Wavelet power spectrum compared to power-law power-spectrum noise [' + conf_level_string + ' conf. level]')
plt.ylabel('period (seconds)')
plt.xlabel('time (seconds)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                          np.ceil(np.log2(period.max())))

plt.yticks(np.log2(Yticks), [str(m) for m in Yticks])
#plt.gca().invert_yaxis()
plt.savefig(os.path.join(outdir, obs + ".red_noise_" + str(slev) + ".png"))
plt.tight_layout()
plt.close("all")

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

