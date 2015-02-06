#
# Program to examine the Fourier power basically pixel by pixel
#
import datatools
import pickle
import os
from matplotlib import pyplot as plt
from timeseries import NewPowerSpectrum
plt.ion()

# input data
corename = '20120923_0000__20120923_0100'
sunlocation = 'disk'
fits_level = '1.5'
wave = '171'
region = 'moss'
window = 'hanning'
manip = 'relative'

# Create the branches in order
branches = [corename, sunlocation, fits_level, wave, region]

# create save and locations
roots = {"pickle": '~/ts/pickle/',
         "image": '~/ts/img/',
         "movie": '~/ts/movies'}
locations = datatools.save_location_calculator(roots, branches)

# Identity tag
ident = datatools.ident_creator(branches)

# Full identity tag
ofilename = '.'.join((ident, window, manip))

# Fourier filename
fourier_filename = 'OUT.' + ofilename + '.fourier_power.pickle'

# Load in the file
loadthispath = os.path.join(locations["pickle"], fourier_filename)
loadthis = open(loadthispath, 'rb')
freqs = pickle.load(loadthis)
pwr = pickle.load(loadthis)
loadthis.close()

ps = NewPowerSpectrum(freqs, pwr[10,10, :])