import os
import numpy as np
from skimage.draw import circle_perimeter
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u


plt.ion()


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
    full_3d_window = np.ones_like(shape)
    for axis, axis_size in enumerate(shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * len(shape)
        filter_shape[axis] = axis_size
        window = filter_function(axis_size).reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        full_3d_window *= window
    return full_3d_window


def k_omega(pwr, cx, cy, method='andres'):
    nk = pwr.shape[0]
    nf = pwr.shape[2]
    answer = np.zeros((nf, nk-1))
    for k in range(1, nk):
        rr, cc = circle_perimeter(cx, cy, k, method=method)
        answer[:, k-1] = np.sum(pwr[rr, cc, :], axis=(0, 1)) / len(rr)
    return answer

# Load the data
filename = 'paper2_six_euv_disk_1.5_171_six_euv.datacube.pkl.fits'

directory = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper2_six_euv/disk/1.5/171/six_euv/"
file_path = os.path.join(directory, filename)
hdulist = fits.read(file_path)
emission_data = hdulist[0].data

# Take the FFT of the windowed data, shift the results so that low frequencies
# are in the centre of the array, and calculate the power
pwr = np.abs(np.fft.fftshift(np.fft.nfft(emission_data * nd_window(emission_data.shape, np.hanning), axes=(0, 1, 2)))) ** 2

# Frequencies
frequencies = np.fft.fftshift(np.fft.fftfreq(emission_data.shape[2], d=12.0))
strictly_positive_frequencies = np.where(frequencies > 0.0)

# Wavenumbers in pixels
wavenumbers = np.fft.fftshift(np.fft.fftfreq(emission_data.shape[1], d=1.0))
zero_wavenumber_index = np.argmin(np.abs(wavenumbers))


# Calculate the k-omega diagram, assuming a square kx, ky plane.
k_om = k_omega(pwr[:, :, strictly_positive_frequencies],
               zero_wavenumber_index,
               zero_wavenumber_index)

# Plot of the k-omega diagram

# Surface plot of the k-omega diagram
