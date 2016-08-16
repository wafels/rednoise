"""
Details of how to handle simulated data and convert it into a format for use
with the red noise processing.
"""

import astropy.units as u

cadence = 8.0 * u.s

header = []

date_obs = '2016-07-21 00:00:00'
center = [0, 0] * u.arcsec
