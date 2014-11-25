import numpy as np
import pickle
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.time import parse_time
import sunpy.cm
import matplotlib.cm as cm
import datetime
import sunpy.net.hek as hek
from matplotlib.collections import PolyCollection

# Movie method
method = 'stills'
#method = 'grab_frame'
#method = 'from_mpl'
obs = 'sunspot'


if obs == 'sunspot':
    filename = '/home/ireland/ts/pickle_cc_final/shutdownfun3_6hr/disk/1.5/171/sunspot/OUT.shutdownfun3_6hr_disk_1.5_171_sunspot.hanning.relative.fft_transform.pickle'
    per = 300.0
    wid = 0.01 / 1000.0

if obs == 'loopfootpoints':
    filename = '/home/ireland/ts/pickle_cc_final/shutdownfun3_6hr/disk/1.5/171/loopfootpoints/OUT.shutdownfun3_6hr_disk_1.5_171_loopfootpoints.hanning.relative.fft_transform.pickle'
    per = 300.0
    wid = 0.10 / 1000.0


# Movie type
filtered = True

nfiles = 200

# Period and frequency width

if obs == "sunspot":
    #
    # Get the sunpot
    #
    client = hek.HEKClient()
    qr = client.query(hek.attrs.Time("2012-09-23 01:00:00", "2012-09-23 02:00:00"), hek.attrs.EventType('SS'))
    p1 = qr[0]["hpc_boundcc"][9: -2]
    p2 = p1.split(',')
    p3 = [v.split(" ") for v in p2]
    p4 = np.asarray([(eval(v[0]), eval(v[1])) for v in p3])
    polygon = np.zeros([1, len(p2), 2])
    polygon[0, :, :] = p4[:, :]
"""
numpoly, numverts = 1, 4
centers = -290 * (np.random.random((numpoly,2)) - 0.5)
offsets = 10 * (np.random.random((numverts,numpoly,2)) - 0.5)
verts = centers + offsets
polygon = np.swapaxes(verts, 0, 1)
"""
# Load the data
pkl_file = open(filename, 'rb')
freqs = pickle.load(pkl_file)
fft_transform = pickle.load(pkl_file)
pkl_file.close()

# set up the filter
npf = freqs.size
filtering = np.zeros(2 * npf + 2)
filtering[0] = 0.0

if filtered:
    filtered_name = 'filtered'
    cm = cm.Blues
    sunspot_color = 'k'
    cen = 1.0 / (1.0 * per)
    const = 1.0 / np.sqrt(2 * np.pi * wid ** 2)
    g = 1.0 * np.exp(-((freqs - cen) ** 2) / (2.0 * wid ** 2))
    for i in range(0, npf):
        filtering[i] = g[i]
        filtering[-i] = g[i]
else:
    filtered_name = 'actual'
    cm = sunpy.cm.cm.sdoaia171
    sunspot_color = 'r'
    filtering[:] = 1.0

# Do the filtering
nx = fft_transform.shape[1]
ny = fft_transform.shape[0]

for i in range(0, ny):
    for j in range(0, nx):
        fft_transform[i, j, :] = fft_transform[i, j, :] * filtering

q = np.sqrt(np.abs(np.fft.ifft(fft_transform)))
nt = q.shape[2]

if nfiles == None:
    nfiles = nt

# Write out a movie
FFMpegWriter = animation.writers['ffmpeg']
fig = plt.figure()
metadata = dict(title='sunspot', artist='Matplotlib', comment="movie for director's seminar")
writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=5000.0)

fname = 'mv.%s.%s.%i' % (obs, filtered_name, per)

#
#
#
if method == 'grab_frame':
    with writer.saving(fig, fname + '.mp4', nt):
        for i in range(0, nt):
            print i, nt
            plt.imshow(np.sqrt(q[:, :, i]))
            writer.grab_frame()
#
#
#
if method == 'from_mpl':
    ims = []
    for i in range(0, nt - 1):
        print i, nt - 1
        ims.append((plt.imshow(q[:, :, i]),))

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)
    im_ani.save(fname + '.mp4', writer=writer)
#
#
# avconv -f image2 -i im.%05d.png -q 0  /home/ireland/Desktop/a.mp4
#
if method == 'stills':
    save = os.path.join('/home/ireland/Desktop/', fname)
    print('Saving to ' + save)
    if not(os.path.isdir(save)):
        os.makedirs(save)
    base_date = parse_time("2012-09-23T00:00:00")
    for i in range(0, nfiles):
        # make the map
        if np.mod(i, 100) == 0:
            print str(i) + ' out of ' + str(nfiles)
        header = {'cdelt1': 0.6, 'cdelt2': 0.6, "crval1": -332.5, "crval2": 17.5,
                  "telescop": 'AIA', "detector": "AIA", "wavelnth": "171",
                  "date-obs": base_date + datetime.timedelta(seconds=12 * i)}
        if filtered:
            ddd = np.sqrt(q[:, :, i])
        else:
            ddd = q[:, :, i]
        my_map = sunpy.map.Map(ddd, header)
        fig, ax = plt.subplots()
        my_map.plot(cmap=cm)
        if obs == 'sunspot':
            coll = PolyCollection(polygon, alpha=1.0, edgecolors=[sunspot_color], facecolors=['none'], linewidth=[5])
            ax.add_collection(coll)
            ax.autoscale_view()
            #for i in range(0,len(polygon)-1):
            #    ax.plot([polygon[0][i][0], polygon[0][i+1][0]], [polygon[0][i][1], polygon[0][i+1][1]])
            #ax.set_ylim(-4, 39)
            #ax.set_xlim(-370, -305)
        plt.savefig(os.path.join(save, 'im.%05i.png' % (i)))
        plt.close('all')

