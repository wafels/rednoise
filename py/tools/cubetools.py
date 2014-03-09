import numpy as np
import sunpy
import matplotlib.pyplot as plt
from scipy.io import readsav
import os
from sunpy.coords.util import rot_hpc
from sunpy.map import Map
from copy import deepcopy
import tsutils
import pickle 

#
# From a directory full of FITS files, return a datacube and a mapcube
#
def get_datacube(path, derotate=False, clip=False):
    """
    Function that goes to a directory and returns a datacube
    """
    if os.path.isfile(path):
        idl = readsav(path)
        return np.swapaxes(np.swapaxes(idl['region_window'], 0, 2), 0, 1)
    else:
        # Get a mapcube
        maps = sunpy.Map(path, cube=True)
        if derotate:
            dc, ysrdisp, xsrdisp = derotated_datacube_from_mapcube(maps, clip=clip)
        else:
            ysrdisp = None
            xsrdisp = None
            nt = len(maps[:])
            ny = maps[0].shape[0]
            nx = maps[0].shape[1]
            dc = np.zeros((ny, nx, nt))
            for i, m in enumerate(maps):
                dc[:, :, i] = m.data[:, :]
        return dc, ysrdisp, xsrdisp, maps


def derotated_datacube_from_mapcube(maps, ref_index=0, clip=False):
    """Return a derotated datacube from a set of maps"""

    # get the dimensions of the datacube
    nt = len(maps[:])
    ny = maps[0].shape[0]
    nx = maps[0].shape[1]

    # reference values
    ref_center = maps[ref_index].center
    ref_time = maps[ref_index].date

    # Output datacube
    datacube = np.zeros((ny, nx, nt))

    # Values of the displacements
    ydiff = np.zeros(nt)
    xdiff = np.zeros(nt)

    # Assume all the maps have the same pointing and the Sun rotates in the
    # field of view.  The center of the field of view at the start time gives
    # the initial field of view that then moves at later times.  This initial
    # location has to be rotated forward to get the correct center of the FOV
    # for the next maps.
    for t, m in enumerate(maps):

        # Store all the data in a 3-d numpy array
        datacube[..., t] = m.data

        # Find the center of the field of view
        newx, newy = rot_hpc(ref_center['x'], ref_center['y'], ref_time,
                                          m.date)

        # Calculate the displacements in tems of pixels
        if newx is None:
            xdiff[t] = 0.0
        else:
            xdiff[t] = (newx - ref_center['x']) / m.scale['x']
        if newy is None:
            ydiff[t] = 0.0
        else:
            ydiff[t] = (newy - ref_center['y']) / m.scale['y']

    # shift the data cube according to the calculated displacements due to
    # solar rotation
    datacube = shift_datacube_layers(datacube, ydiff, xdiff)

    # Optionally clip the datacube to remove data that may be affected by edge
    # effects due to solar derotation.
    if clip:
        return clip_edges(datacube, ydiff, xdiff), ydiff, xdiff
    else:
        return datacube


def get_datacube_from_mapcube(mapcube):
    """
    Extract the image data from a mapcube.
    """
    # get the dimensions of the datacube
    nt = len(mapcube[:])
    ny = mapcube[0].shape[0]
    nx = mapcube[0].shape[1]

    # Output datacube
    datacube = np.zeros((ny, nx, nt))
    for t, m in enumerate(mapcube):

        # Store all the data in a 3-d numpy array
        datacube[..., t] = m.data

    # Return the datacube
    return datacube


def get_properties_from_mapcube(mapcube):
    """
    Extract the mapcube properties out of the mapcube.
    """
    pass


def visualize(datacube, delay=0.1):
    """
    Visualizes a datacube.  Requires matplotlib 1.1
    """
    nt = datacube.shape[2]
    print nt
    fig = plt.figure()

    axes = fig.add_subplot(111)
    axes.set_xlabel('X-position')
    axes.set_ylabel('Y-position')

    #extent = wave_maps[0].xrange + wave_maps[0].yrange
    #axes.set_title("%s %s" % (wave_maps[0].name, wave_maps[0].date))

    img = axes.imshow(datacube[:, :, 0], origin='lower')
    #fig.colorbar(img)
    fig.show()

    for z in np.arange(0, nt):
        m = datacube[:, :, z]
        axes.set_title("%i" % (z))
        img.set_data(m)
        plt.pause(delay)
    return None


def sum_over_space(dc, remove_mean=False):
    """
    Sum a datacube over all the spatial locations
    """
    # Get some properties of the datacube
    ny = dc.shape[0]
    nx = dc.shape[1]
    nt = dc.shape[2]

    # Storage for the
    fd = np.zeros((nt))
    for i in range(0, nx):
        for j in range(0, ny):
            d = dc[j, i, :].flatten()

            # Fix the data for any non-finite entries
            d = tsutils.fix_nonfinite(d)

            # Remove the mean
            if remove_mean:
                d = d - np.mean(d)

            # Sum up all the data
            fd = fd + d
    return fd


# Data that we want to save
def save_output(filename, datacube, times, pixel_index):
    outputfile = open(filename, 'wb')
    pickle.dump(datacube, outputfile)
    pickle.dump(times, outputfile)
    pickle.dump(pixel_index, outputfile)
    outputfile.close()
    print('Saved to ' + filename)
    return


# Save multiple regions
def save_region(dc, output, regions, wave, times):
    keys = regions.keys()
    for region in keys:
        pixel_index = regions[region]
        y = pixel_index[0]
        x = pixel_index[1]
        #filename = region + '.' + wave + '.' + str(pixel_index) + '.datacube.pickle'
        filename = region + '.' + wave + '.datacube.pickle'
        save_output(os.path.join(output, filename),
                    dc[y[0]: y[1], x[0]:x[1], :],
                    times,
                    pixel_index)


def makedirs(output):
    if not os.path.isdir(output):
        os.makedirs(output)


def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for
    the 1d description.  Copied from http://code.google.com/p/agpy/source/browse/trunk/agpy/psds.py?r=343
    """
    # scalar unity; don't window if dims are too small
    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        return np.hanning(N)
    else:
        return np.outer(np.hanning(M), np.hanning(N))

