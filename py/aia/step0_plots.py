#
# Utilities to plot out details of step 1 of the analysis
#
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import details_study as ds

layer_index_prop = {"label": "ref. layer index",
                    "color": "k",
                    "linestyle": "--"}

zero_prop = {"label": None,
             "color": "k",
             "linestyle": ":"}

shift_prop = {"x": {"label": "x",
                    "color": "b",
                    "linestyle": '-'},
              "y": {"label": "y",
                    "color": "g",
                    "linestyle": '-'}}


def plot_shifts(shifts, title, layer_index,
                unit='arcsec',
                filepath=None,
                x=None,
                xlabel='mapcube layer index'):
    """
    Plot out the shifts used to move layers in a mapcube
    :param shifts:
    :param title:
    :param layer_index:
    :param unit:
    :return:
    """
    if x is None:
        xx = np.arange(0, len(shifts['x']))
    else:
        xx = x

    plt.close('all')
    for c in ['x', 'y']:
        plt.plot(xx, shifts[c].to(unit),
                 label=shift_prop[c]['label'],
                 color=shift_prop[c]['color'],
                 linestyle=shift_prop[c]['linestyle'])

    plt.axvline(xx[layer_index],
                label=layer_index_prop['label'] + ' (%i)' % layer_index,
                color=layer_index_prop['color'],
                linestyle=layer_index_prop['linestyle'])
    plt.axhline(0.0,
                label=zero_prop['label'],
                color=zero_prop['color'],
                linestyle=zero_prop['linestyle'])
    plt.legend(framealpha=0.5)
    nlayers = ' (%i maps)' % len(shifts['x'])
    plt.xlabel(xlabel + nlayers)
    plt.ylabel(unit)
    plt.title(title)
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None

"""
def cc_shift_analysis(cc_shifts, x_scale, y_scale, layer_index,
                      file_path='~', dt=12.0*u.s, filename=None):
    # Convert to the required units (typically pixels)
    ccx = cc_shifts['x'] / x_scale
    ccy = cc_shifts['y'] / y_scale

    # Units
    displacement_unit = ccx.unit

    # Calculate the net displacement
    displacement = np.sqrt(ccx.value ** 2 + ccy.value ** 2)
    npwr = len(displacement)

    # Apodization window
    window = np.hanning(npwr)

    # FFT power
    pwr_r = np.abs(np.fft.fft(displacement * window, norm='ortho')) ** 2
    pwr_ccx = np.abs(np.fft.fft(ccx * window, norm='ortho')) ** 2
    pwr_ccy = np.abs(np.fft.fft(ccy * window, norm='ortho')) ** 2

    # FFT frequency
    freq = np.fft.fftfreq(npwr, dt.to(u.s).value)

    # Make the plot
    plt.close('all')
    plt.figure(1)
    plt.semilogy(freq[0:npwr // 2], pwr_r[0:npwr // 2], label='net displacement')
    plt.semilogy(freq[0:npwr // 2], pwr_ccx[0:npwr // 2], label='x displacement')
    plt.semilogy(freq[0:npwr // 2], pwr_ccy[0:npwr // 2], label='y displacement')
    plt.axis('tight')
    plt.grid('on')
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'power ({:s})'.format((displacement_unit ** 2)._repr_latex_()))
    title = 'FFT power of cross-correlation displacement\n'
    analysis_title = 'AIA {:s}, {:n} images, FITS level={:s}\n'.format(ds.wave,
                                                                       npwr,
                                                                       ds.fits_level)
    analysis_title += 'derotation and cross-correlation image index={:n}'.format(layer_index)
    plt.title(title + analysis_title)
    plt.legend(framealpha=0.5)
    plt.tight_layout()

    # Save the image
    full_file_path = os.path.join(os.path.expanduser(file_path), filename)
    plt.savefig(full_file_path)


def map_times(dts, layer_index_date, file_path='~', filename=None):
    analysis_title = 'AIA {:s}, {:n} images, FITS level={:s}\n'.format(ds.wave,
                                                                       len(dts),
                                                                       ds.fits_level)
    plt.figure(2)
    plt.plot(np.arange(len(dts)), dts)
    plt.axis('tight')
    plt.axvline(layer_index, color='r', label='layer index')
    plt.axhline(0.0, color='k', linestyle=":", label='reference layer')
    plt.xlabel('sample number')
    plt.ylabel('time (s) rel. to derotation and cross-correlation image\n at {:s}'.format(str(layer_index_date)))
    title = 'FITS recorded observation time from initial observation\n'
    plt.title(title + analysis_title)
    plt.legend(framealpha=0.5)
    plt.tight_layout()

    # Save the image
    full_file_path = os.path.join(os.path.expanduser(file_path), filename)
    plt.savefig(full_file_path)
"""
