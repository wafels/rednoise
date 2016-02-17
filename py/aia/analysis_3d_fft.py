import numpy as np
import os
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u

plt.ion()
filename = '/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper2_six_euv/disk/1.5/171/six_euv/paper2_six_euv_disk_1.5_171_six_euv.datacube.pkl.fits'


nspace = 333
nt = 1800

hdulist = fits.open(filename)


def apodize3d(ny, nx, nt, func=np.hanning):

    apodize_1d = func(nspace)
    apodize_x = np.tile(apodize_1d, (nspace, 1))
    apodize_y = np.transpose(apodize_x)
    apodize_space = apodize_x * apodize_y
    apodize_time = func(nt)

    apodize = np.zeros_like(hdulist[0].data[:, 0:333, :])
    for i in range(0, nt):
        apodize[:, :, i] = apodize_time[i] * apodize_space

    return apodize

apodize = apodize3d(nspace, nspace, nt)

ft = np.abs(np.fft.fftn(hdulist[0].data[0:nspace, 0:nspace, 0:nt] * apodize,
                        axes=(0, 1, 2))) ** 2

cadence = 12 * u.s
scale = 0.6 * u.arcsecond / u.pix


freq = np.fft.freq(1800, cadence.to('s').value) / u.s

wavenumber = np.fft.freq(333, scale.value) / scale.unit


