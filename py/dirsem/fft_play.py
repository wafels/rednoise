import numpy as np
import pickle
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Movie method
method = 'stills'
#method = 'grab_frame'
#method = 'from_mpl'

# Movie type
filtered = True

# Period and frequency width
per = 300
wid = 0.01 / 1000.0

# Load the data
filename = '/home/ireland/ts/pickle_cc_final/shutdownfun3_6hr/disk/1.5/171/sunspot/OUT.shutdownfun3_6hr_disk_1.5_171_sunspot.hanning.relative.fft_transform.pickle'
pkl_file = open(filename, 'rb')
freqs = pickle.load(pkl_file)
fft_transform = pickle.load(pkl_file)
pkl_file.close()

# set up the filter
npf = freqs.size
filtering = np.zeros(2 *npf + 2)
filtering[0] = 0.0

if filtered:
    filtered_name = 'filtered'
    cen = 1.0/(1.0 * per)
    const = 1.0/np.sqrt(2*np.pi*wid**2)
    g = 1.0 * np.exp( -((freqs - cen) ** 2)/(2.0 * wid ** 2))
    for i in range(0, npf):
        filtering[i] = g[i]
        filtering[-i] = g[i]
else:
    filtered_name = 'actual'
    filtering[:] = 1.0

# Do the filtering
nx = fft_transform.shape[1]
ny = fft_transform.shape[0]

for i in range(0, ny):
    for j in range(0, nx):
        fft_transform[i, j, :] = fft_transform[i, j, :] * filtering

q = np.sqrt(np.abs(np.fft.ifft( fft_transform )))
nt = q.shape[2]

# Write out a movie
FFMpegWriter = animation.writers['ffmpeg']
fig = plt.figure()
metadata = dict(title='sunspot', artist='Matplotlib', comment="movie for director's seminar")
writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=5000.0)

fname = 'mv.%s.%i' % (filtered_name, per)

#
#
#
if method == 'grab_frame':
    with writer.saving(fig, 'output_movie.sunspt.mp4', nt):
        for i in range(0, nt):
            print i,nt
            plt.imshow(np.sqrt(q[:, :, i * reduction]))
            writer.grab_frame()
#
#
#
if method == 'from_mpl':
    ims=[]
    for i in range(0, nt-1):
        print i, nt-1
        ims.append((plt.imshow(q[:, :, i]),))

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)
    im_ani.save(fname + '.mp4', writer=writer)
#
#
#
if method == 'stills':
    save = '/home/ireland/Desktop/movie%i/' % (per)
    for i in range(0, nt - 1):
        plt.imshow(np.sqrt(q[:, :, i]))
        plt.title('time = %i seconds' % (i * 12) )
        plt.savefig(save + 'im.%05i.png' % (i))

