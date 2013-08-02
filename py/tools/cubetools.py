import numpy as np
import scipy
import sunpy
import matplotlib.pyplot as plt


def derotated_datacube_from_mapcube(maps, ref_index=0, diff_limit=0.01):
    """Return a datacube from a set of maps"""

    # get the dimensions of the datacube
    nt = len(maps[:])
    ny = maps[0].shape[0]
    nx = maps[0].shape[1]

    # reference values
    ref_center = maps[ref_index].center
    ref_time = maps[ref_index].date

    # Output datacube
    datacube = np.zeros((ny, nx, nt))

    # maximum differences found si far
    max_abs_xdiff = 0.0
    max_abs_ydiff = 0.0

    # Assume all the maps have the same pointing and the Sun rotates in the
    # field of view.  The center of the field of view at the start time gives
    # the initial field of view that then moves at later times.  This initial
    # location has to be rotated forward to get the correct center of the FOV
    # for the next maps.
    for t, m in enumerate(maps):
        newx, newy = sunpy.coords.rot_hpc(ref_center['x'],
                                          ref_center['y'],
                                          ref_time,
                                          m.date)
        if newx is None:
            xdiff = 0
        else:
            xdiff = (newx - ref_center['x']) / m.scale['x']
        if newy is None:
            ydiff = 0
        else:
            ydiff = (newy - ref_center['y']) / m.scale['y']

        if np.abs(xdiff) > max_abs_xdiff:
            max_abs_xdiff = np.ceil(np.abs(xdiff))

        if np.abs(ydiff) > max_abs_ydiff:
            max_abs_ydiff = np.ceil(np.abs(ydiff))

        #print xdiff, ydiff
        # In matplotlib, the origin of the image is in the top right of the
        # the array.  Therefore, shifting in the y direction has to be reversed
        # Also, the first dimension in the numpy array refers to what a solar
        # physicist refers to as the north-south (i.e., 'y') direction.
        #print t, xdiff, ydiff
        if np.abs(ydiff) >= diff_limit and np.abs(xdiff) >= diff_limit:
            shifted = scipy.ndimage.interpolation.shift(m.data, [-ydiff, -xdiff])
        else:
            shifted = m.data
        # Store all the shifted data.
        datacube[..., t] = shifted
        #print t, m.date, maps[t].center, xdiff, ydiff

    # Zero out the edges where the data has been moved
    return datacube[0:ny - max_abs_ydiff - 1, 0:nx - max_abs_xdiff - 1:, :]


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

    img = axes.imshow(dc[:, :, 0], origin='lower')
    #fig.colorbar(img)
    fig.show()

    for z in np.arange(1, nt - 1):
        m = datacube[:, :, z]
        #axes.set_title("%i" % (i))
        img.set_data(m)
        plt.pause(delay)
    return None
