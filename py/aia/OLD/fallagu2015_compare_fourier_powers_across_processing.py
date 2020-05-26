import numpy as np
import os
from matplotlib import rc_file
matplotlib_file = '~/ts/rednoise/py/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import matplotlib.pyplot as plt

from details_plots import log_10_product, five_minutes, three_minutes

plt.ion()

data = {"regular 1.5": {"filepath": "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper2_six_euv/disk/1.5/171/six_euv/paper2_six_euv_disk_1.5_171_six_euv.datacube.hanning.sum_log_fft_power_relative_intensities.npz"},
        "PSF removed": {"filepath": "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_PSF/disk/1.5/171/six_euv/paper3_PSF_disk_1.5_171_six_euv.datacube.hanning.sum_log_fft_power_relative_intensities.npz"},
        "PSF removed + BM3D": {"filepath": "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_BM3D/disk/1.5/171/six_euv/paper3_BM3D_disk_1.5_171_six_euv.datacube.hanning.sum_log_fft_power_relative_intensities.npz"},
        "PSF removed + BLSGSM": {"filepath": "/home/ireland/ts/pickle/cc_True_dr_True_bcc_False/paper3_BLSGSM/disk/1.5/171/six_euv/paper3_BLSGSM_disk_1.5_171_six_euv.datacube.hanning.sum_log_fft_power_relative_intensities.npz"}}
keys = data.keys()

plt.close('all')

ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
fz = 'mHz'

xformatter = plt.FuncFormatter(log_10_product)
ax.xaxis.set_major_formatter(xformatter)


pkeep = {}

for key in keys:
    z = np.load(data[key]["filepath"])

    f = np.log10(1000 * z["pfrequencies"])
    p = z["drel_power"]

    ax.plot(10.0**f, 10.0**p, label=key, linewidth=3.0)

    pkeep[key] = p


plt.axvline((1.0/five_minutes.position).to(fz).value,
            color=five_minutes.color,
            label=five_minutes.label,
            linestyle=five_minutes.linestyle,
            linewidth=five_minutes.linewidth)

plt.axvline((1.0/three_minutes.position).to(fz).value,
            color=three_minutes.color,
            label=three_minutes.label,
            linestyle=three_minutes.linestyle,
            linewidth=three_minutes.linewidth)

plt.title('AIA 171$\AA$: average logarithm of power for different processing')
plt.xlabel(r'frequency $\nu$ (%s)' % fz)
plt.ylabel(r"average $\log_{10}$(relative power)")
plt.legend(framealpha=0.5)
plt.show()
#plt.savefig('/home/ireland/Desktop/compare.png')

