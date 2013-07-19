"""
Co-align a set of maps
"""

# Test 1
import sunpy
import scipy
import numpy as np

directory = '/home/ireland/Data/AIA_Data/test1/'

maps = sunpy.make_map(directory)

ref_index = 0
ref_center = maps[ref_index].center
ref_time = maps[ref_index].time


nt = len(maps)
ny = maps[0].shape[0]
nx = maps[0].shape[1]

datacube = np.zeros((ny, nx, nt))

max_abs_xdiff = 0.0
max_abs_ydiff = 0.0

for m, t in enumerate(maps):
    # Assume all the maps have the same pointing and the Sun rotates in the
    # field of view.  The center of the field of view at the start time gives
    # the initial field of view that then moves at later times.  This initial
    # location has to be rotated forward to get the correct center of the FOV
    # for the next maps.
    newx, newy = sunpy.coords.rot_hpc(ref_center['x'],
                                      ref_center['y'],
                                      ref_time,
                                      m.time)
    if newx is None:
        xdiff = 0
    else:
        xdiff = newx - ref_center['x']
    if newy is None:
        ydiff = 0
    else:
        ydiff = newy - ref_center['y']

    if np.abs(xdiff) > max_abs_xdiff:
        max_abs_xdiff = np.ceil(np.abs(xdiff))

    if np.abs(ydiff) > max_abs_ydiff:
        max_abs_ydiff = np.ceil(np.abs(ydiff))

    # In matplotlib, the origin of the image is in the top right of the
    # the array.  Therefore, shifting in the y direction has to be reversed.
    # Also, the first dimension in the numpy array refers to what a solar
    # physicist refers to as the north-south (i.e., 'y') direction.
    shifted = scipy.ndimage.interpolation.shift(m, [-ydiff, xdiff])
    # Store all the shifted data.
    datacube[:, :, t] = shifted[:, :]

# Zero out the edges where the data has been moved
datacube[:-max_abs_ydiff, :, :] = 0.0
datacube[:, :-max_abs_xdiff, :] = 0.0