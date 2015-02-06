#
# test solar derotation functions
#
import os
import matplotlib.pyplot as plt
from sunpy.map import Map
from sunpy.physics.transforms.solar_rotation import calculate_solar_rotate_shift, mapcube_solar_derotate
from sunpy.image.coalignment import mapcube_coalign_by_match_template
plt.ion()

# Where the data is
dir = os.path.expanduser('~/Data/ts/quickrequest/171')

# Files
l = sorted(os.listdir(dir))

# Full path
fp = [os.path.join(dir, f) for f in l]

# Create a Mapcube
mc = Map(fp, cube=True)


shifts = calculate_solar_rotate_shift(mc)

nmc = mapcube_solar_derotate(mc, clip=False)

nmc.peek()

cmc = mapcube_coalign_by_match_template(mc)
