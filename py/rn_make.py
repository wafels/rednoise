
import numpy as np

n = 86400
dt = 12.0

# White noise
wn = np.random.normal(size=(n,))

# FFT of the white noise
wn_fft = np.fft.rfft(wn)

# 
f = np.fft.fftfreq(n, dt)[:len(wn_fft)]
f[-1]=np.abs(f[-1])

fft_sim = wn_fft[1:] * f[1:]**-1
T_sim = np.fft.irfft(fft_sim)


