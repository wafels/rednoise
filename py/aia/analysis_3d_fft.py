import os
import cPickle as pickle
import numpy as np
from skimage.draw import circle_perimeter
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import astropy.units as u

from details_plots import log_10_product

plt.ion()

#choice = 'test'
choice = 'BM4D'
#choice = 'BM3D'
#choice = 'PSF_removed'
#choice = 'no_denoise'

log_x_axis = True  # Default is False
log_y_axis = True  # Default is True

root_directory = os.path.expanduser('~/ts/noise_reduction')  # Where to dump the output


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


@u.quantity_input(k0x_index=u.dimensionless_unscaled, k0y_index=u.dimensionless_unscaled, k=u.dimensionless_unscaled)
def k_omega_pixel_circle(k0x_index, k0y_index, k, method='andres'):
    """
    Calculate which pixels lie at on a circle of radius 'k' from the center
    'cx, cy'.  This can be used to calculate

    Parameters
    ----------
    k0x_index : integer
        pixel index in the x direction of the (kx, ky, omega) array of where
        the kx wavenumber equals zero

    k0y_index : integer
        pixel index in the y direction of the (kx, ky, omega) array of where
        the ky wavenumber equals zero

    k : integer
        radius of the circle in pixels

    method : 'andres' | 'bresenham'
        the method used to calculate the which pixels

    Returns
    -------
    The x, y locations of pixels on the circle in the kx, ky plane.

    """
    rr, cc = circle_perimeter(int(k0x_index.to(u.dimensionless_unscaled).value),
                              int(k0y_index.to(u.dimensionless_unscaled).value),
                              int(k.to(u.dimensionless_unscaled).value),
                              method=method)
    return rr*u.dimensionless_unscaled, cc*u.dimensionless_unscaled


@u.quantity_input(k0x_index=u.dimensionless_unscaled, k0y_index=u.dimensionless_unscaled, kmin=u.dimensionless_unscaled, kmax=u.dimensionless_unscaled)
def k_omega_pixel_circles(k0x_index, k0y_index, kmin, kmax, method='andres'):
    """
    Calculate a list of circles centered at (k0x_index, k0y_index) in the
    kx, ky plane that have radii in the (closed) range kmin to kmax.

    Parameters
    ----------
    k0x_index : integer
        index in the x direction of the (kx, ky, omega) array of where the
        kx wavenumber equals zero

    k0y_index : integer
        index in the y direction of the (kx, ky, omega) array of where the
        ky wavenumber equals zero

    kmin : integer
        minimum radius of the circle

    kmax : integer
        maximum radius of the circle

    method : 'andres' | 'bresenham'
        the method used to calculate the which pixels

    Returns
    -------
    An ordered list of circles.

    """
    answer = []
    for k in range(int(kmin.to(u.dimensionless_unscaled).value), int(kmax.to(u.dimensionless_unscaled).value) + 1):
        rr, cc = k_omega_pixel_circle(k0x_index,
                                      k0y_index,
                                      k*u.dimensionless_unscaled,
                                      method=method)
        answer.append([rr, cc])
    return answer


def k_omega_power(pwr, circles):
    """
    Calculate the k-omega plot for an input three dimensional FFT power of the
    form (kx, ky, omega).

    Parameters
    ----------
    pwr : 3-d numpy.ndarray
        An array of FFT power of the form (nkx, nky, nf).

    circles : list
        A list of circles.  The k-omega plot is calculated by averaging the
        power at the pixel locations in the kx, ky plane indicated by each
        circle.  The same integral is performed for all frequencies.

    Returns
    -------
    answer : 2-d numpy.ndarray
        The k-omega diagram of the input three-dimensional FFT.
    """
    answer = np.zeros((pwr.shape[2], len(circles)))
    for k, px in enumerate(circles):
        rr = np.asarray(px[0].to(u.dimensionless_unscaled).value, dtype=np.int)
        cc = np.asarray(px[1].to(u.dimensionless_unscaled).value, dtype=np.int)
        answer[:, k] = np.sum(pwr[rr[:], cc[:], :], axis=0) / len(rr)
    return answer


# Load the data
if choice == "no_denoise":
    filename = 'paper2_six_euv_disk_1.5_171_six_euv.datacube.1-4.pkl.fits'
    directory = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper2_six_euv/disk/1.5/171/six_euv/"
    scale = 0.6 * u.arcsec / u.pix

if choice == "PSF_removed":
    filename = 'paper3_PSF_disk_1.5_171_six_euv.datacube.1-5.pkl.fits'
    directory = "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_PSF/disk/1.5/171/six_euv/"
    scale = 0.6 * u.arcsec / u.pix

if choice == "BM3D":
    filename = 'paper3_BM3D_disk_1.5_171_six_euv.datacube.1-11.pkl.fits'
    directory = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_BM3D/disk/1.5/171/six_euv/'
    scale = 0.6 * u.arcsec / u.pix

if choice == "BM4D":
    filename = 'AIA.171.2012_09_23.BM4DProcess.1-11.fits'
    directory = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_BM4D/disk/1.5/171/six_euv/'
    scale = 0.6 * u.arcsec / u.pix

if choice != "test":
    file_path = os.path.join(directory, filename)
    print('Reading %s' % file_path)
    hdulist = fits.open(file_path)
    emission_data = hdulist[0].data
else:
    emission_data = np.random.random((50, 60, 100))
    scale = 1 * u.dimensionless_unscaled


# Minimum spatial extent - this constrains the number of wavenumbers since
# we are only dealing with square spatial extents
nspace = np.min([emission_data.shape[0], emission_data.shape[1]])

# Emission data is analyzed as relative change intensity
emission_data = emission_data[0:nspace, 0:nspace, :]
emission_data_mean = np.mean(emission_data, axis=2, keepdims=True)
emission_data = (emission_data - emission_data_mean)/emission_data_mean
print('Emission data shape ', emission_data.shape)

# Take the FFT of the windowed data, shift the results so that low frequencies
# are in the centre of the array, and calculate the power
pwr = np.abs(np.fft.fftshift(np.fft.fftn(emission_data * nd_window((nspace, nspace, emission_data.shape[2]), np.hanning), axes=(0, 1, 2)))) ** 2

# Frequencies
frequency_unit = 'mHz'
frequencies = np.fft.fftshift(np.fft.fftfreq(emission_data.shape[2], d=12.0))
strictly_positive_frequencies = np.asarray(np.where(frequencies > 0.0))
spm = (frequencies[strictly_positive_frequencies] / u.s).to(frequency_unit)

# Wavenumbers in pixels
wavenumbers = np.fft.fftshift(np.fft.fftfreq(nspace, d=scale.value)) / (scale.unit * u.pix)
zero_wavenumber_index = np.argmin(np.abs(wavenumbers.value)) * u.dimensionless_unscaled
wn = wavenumbers[zero_wavenumber_index.to(u.dimensionless_unscaled).value + 1:]

# Calculate the circles in the k-omega plane we will use to do the integrals
kmin = 1 * u.dimensionless_unscaled
kmax = pwr.shape[0]*u.dimensionless_unscaled - zero_wavenumber_index - 1*u.dimensionless_unscaled
print('Wavenumber index range ', kmin, kmax)
print('Frequency range ', spm[0, 0], spm[0, -1])
circles = k_omega_pixel_circles(zero_wavenumber_index,
                                zero_wavenumber_index,
                                kmin,
                                kmax)

# Calculate the k-omega diagram, assuming a square kx, ky plane.
log10_k_om = np.log10(k_omega_power(pwr[:, :, strictly_positive_frequencies[0, :]], circles))


"""
# Plot of the k-omega diagram
plt.imshow(log10_power, aspect='auto', cmap=cm.Set1, origin='lower',
           extent=(np.log10(wn[0].value), np.log10(wn[-1].value), spm[0, 0].value, spm[0, -1].value))
plt.xlabel('wavenumber (%s)' % str(wn.unit))
plt.ylabel('frequency (%s)' % str(spm.unit))
plt.colorbar()
"""
print(np.min(log10_k_om), np.max(log10_k_om))
five_minutes = (1.0 / 300.0) / u.s
three_minutes = (1.0 / 180.0) / u.s

# Plot of the k-omega diagram with logarithmic frequency.
# lowest value = 6.848, highest value = 20.12
fig, ax = plt.subplots()
if log_y_axis:
    yformatter = plt.FuncFormatter(log_10_product)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(yformatter)
if log_x_axis:
    xformatter = plt.FuncFormatter(log_10_product)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(xformatter)
cmap = cm.nipy_spectral
cax = ax.pcolormesh(wn.value, spm[0, :].value, log10_k_om, cmap=cmap, vmin=0.84, vmax=11.96)
wn_min = '%7.4f' % wn[0].value
wn_max = '%7.4f' % wn[-1].value
f_min = '%7.4f' % spm[0, 0].to(frequency_unit).value
f_max = '%7.4f' % spm[0, -1].to(frequency_unit).value
wavenumber_label = r'wavenumber (%s) [range=%s$\rightarrow$%s]' % (str(wn.unit), wn_min, wn_max)
frequency_label = r'frequency (%s) [range=%s$\rightarrow$%s]' % (str(spm.unit), f_min, f_max)
ax.set_xlabel(wavenumber_label)
ax.set_ylabel(frequency_label)
ax.set_xlim(wn[0].value, wn[-1].value)
ax.set_ylim(spm[0, 0].value, spm[0, -1].value)
ax.set_title(r"%s [power=%4.2f$\rightarrow$%4.2f]" % (choice, log10_k_om.min(), log10_k_om.max()))
f5 = ax.axhline(five_minutes.to(frequency_unit).value, linestyle='--', color='k')
f3 = ax.axhline(three_minutes.to(frequency_unit).value, linestyle='-.', color='k')
fig.colorbar(cax, label=r'$\log_{10}(power)$')
ax.legend((f3, f5), ('three minutes', 'five minutes'), fontsize=8.0, framealpha=0.5)
fig.tight_layout()

output_filename = 'analysis_3d_fft.{:s}.logx={:s}.logy={:s}.{:s}.png'.format(choice,
                                                             str(log_x_axis),
                                                             str(log_y_axis),
                                                             cmap.name)
file_path = os.path.join(root_directory, output_filename)
fig.set_size_inches(24, 16)
fig.savefig(file_path)

# Save the k_omega data
file_path = os.path.join(root_directory, output_filename + '.pkl')
f = open(file_path, 'wb')
pickle.dump(10.0**log10_k_om, f)
f.close()

#
# Mean powers
#
for axis, data_type in enumerate(('wavenumber', 'frequency')):

    if axis == 0:
        xaxis = wn.value
        xlabel = wavenumber_label
    else:
        xaxis = spm[0, :].to(frequency_unit).value
        xlabel = frequency_label

    for mean_style, mean_style_label in enumerate(('mean Fourier power', 'mean log10(Fourier power)')):
        if mean_style == 0:
            mean_power = np.mean(10.0**log10_k_om, axis=axis)
        else:
            mean_power = 10.00*np.mean(log10_k_om, axis=axis)

        # Plot mean power
        plt.close('all')
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(log_10_product))
        ax.set_xscale('log')
        ax.set_xlabel(xlabel)

        ax.set_yscale('log')
        ax.set_ylabel('Fourier power')

        ax.plot(xaxis, mean_power)
        ax.set_title(choice + '\n{:s}'.format(mean_style_label))
        fig.tight_layout()
        fig.savefig(file_path + '.mean_power_{:s}.{:s}.png'.format(mean_style_label, data_type))




