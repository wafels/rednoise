import os
import numpy as np
from skimage.draw import circle_perimeter
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import astropy.units as u


plt.ion()

choice = 'BM4D'
#choice = 'BM3D'
#choice = 'no_denoise'


def nd_window(shape, filter_function):
    """
    Define an n-dimensional window function based on one dimensional functions.

    Parameters
    ----------
    shape : list
            The shape of the data array.  The window function will have the
            same size.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """
    full_3d_window = np.ones(shape)
    for axis, axis_size in enumerate(shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * len(shape)
        filter_shape[axis] = axis_size
        window = filter_function(axis_size).reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        full_3d_window *= window
    return full_3d_window


def k_omega(pwr, cx, cy, method='andres'):
    """
    Calculate a k-omega plot from 3-d FFT results
    :param pwr: an input three-dimensional FFT power array of the form (nk, nk, nf)
     where nk is the
    :param cx:
    :param cy:
    :param method:
    :return:
    """
    nk_strictly_positive = pwr.shape[0] - cx - 1
    nf = pwr.shape[2]
    answer = np.zeros((nf, nk_strictly_positive))
    for k in range(0, nk_strictly_positive):
        rr, cc = circle_perimeter(cx, cy, k, method=method)
        answer[:, k] = np.sum(pwr[rr[:], cc[:], :], axis=0) / len(rr)
    return answer

# Load the data
if choice == "no_denoise":
    filename = 'paper2_six_euv_disk_1.5_171_six_euv.datacube.pkl.fits'
    directory = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper2_six_euv/disk/1.5/171/six_euv/"

if choice == "BM3D":
    filename = 'paper3_BM3D_disk_1.5_171_six_euv.datacube.pkl.fits'
    directory = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_BM3D/disk/1.5/171/six_euv/'

if choice == "BM4D":
    filename = 'paper3_BM4D_disk_1.5_171_six_euv.datacube.pkl.fits'
    directory = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_BM4D/disk/1.5/171/six_euv/'

file_path = os.path.join(directory, filename)
print('Reading %s' % file_path)
hdulist = fits.open(file_path)
emission_data_shape = hdulist[0].data.shape
nspace = np.min([emission_data_shape[0], emission_data_shape[1]])
emission_data = hdulist[0].data[0:nspace, 0:nspace, :]
emission_data_mean = np.mean(emission_data, axis=2, keepdims=True)
emission_data = (emission_data - emission_data_mean)/emission_data_mean

# Take the FFT of the windowed data, shift the results so that low frequencies
# are in the centre of the array, and calculate the power
pwr = np.abs(np.fft.fftshift(np.fft.fftn(emission_data * nd_window((nspace, nspace, emission_data.shape[2]), np.hanning), axes=(0, 1, 2)))) ** 2

# Frequencies
frequencies = np.fft.fftshift(np.fft.fftfreq(emission_data.shape[2], d=12.0))
strictly_positive_frequencies = np.asarray(np.where(frequencies > 0.0))
spm = (frequencies[strictly_positive_frequencies] / u.s).to('mHz')

# Wavenumbers in pixels
wavenumbers = np.fft.fftshift(np.fft.fftfreq(nspace, d=1.0))
zero_wavenumber_index = np.argmin(np.abs(wavenumbers))
wn = wavenumbers[zero_wavenumber_index + 1:] / u.pix


# Calculate the k-omega diagram, assuming a square kx, ky plane.
k_om = k_omega(pwr[:, :, strictly_positive_frequencies[0, :]],
               zero_wavenumber_index,
               zero_wavenumber_index)

log10_power = np.log10(k_om)

"""
# Plot of the k-omega diagram
plt.imshow(log10_power, aspect='auto', cmap=cm.Set1, origin='lower',
           extent=(np.log10(wn[0].value), np.log10(wn[-1].value), spm[0, 0].value, spm[0, -1].value))
plt.xlabel('wavenumber (%s)' % str(wn.unit))
plt.ylabel('frequency (%s)' % str(spm.unit))
plt.colorbar()
"""
print(np.min(np.log10(k_om)), np.max(np.log10(k_om)))

# Plot of the k-omega diagram with logarithmic frequency.
# lowest value = 6.848, highest value = 20.12
fig, ax = plt.subplots()
ax.set_yscale('log')
#ax.set_xscale('log')
cax = ax.pcolor(wn.value, spm[0, :].value, log10_power, cmap=cm.nipy_spectral, vmin=0.84, vmax=11.96)
ax.set_xlabel('wavenumber (%s)' % str(wn.unit))
ax.set_ylabel('frequency (%s)' % str(spm.unit))
ax.set_xlim(wn[0].value, wn[-1].value)
ax.set_ylim(spm[0, 0].value, spm[0, -1].value)
ax.set_title("%s, min=%4.2f, max=%4.2f" % (choice, log10_power.min(), log10_power.max()))
fig.colorbar(cax)
