#
# Some magic numbers and parameters used in plots
#
import os
import numpy as np
import astropy.units as u
import details_study as ds


# Units that frequencies will be plotted in
fz = 'mHz'

# Three minute oscillation
three_minutes = 180 * u.s

# Five minute oscillation
five_minutes = 300 * u.s


# Histograms have the following type
hloc = (100,)  # 'scott', 'knuth', 'freedman')

#
# Useful strings for plots
#
# Reduced chi-squared
rchi2s = '$\chi^{2}_{r}$'
rchi2string = '$<$' + rchi2s + '$<$'

# Reduced chi-squared limit colors
rchi2limitcolor = ['r', 'y']


# Create a label which shows the limits of the reduced chi-squared value. Also
# define some colors that signify the upper and lower levels of the reduced
# chi-squared/
def rchi2label(rchi2limit):
    return '%1.2f%s%1.2f' % (rchi2limit[0], rchi2string, rchi2limit[1])

#
# Probability string that defines a percentage output
#
def percentstring(pvalue):
    return '%2.1f%%<p<%2.1f%%' % (100 * pvalue[0], 100 * pvalue[1])


def percent_lo(pvalue):
    return '%s (p$<$%2.1f%%)' % (rchi2s, 100 * pvalue[0])


def percent_hi(pvalue):
    return '%s (p$>$%2.1f%%)' % (rchi2s, 100 - 100 * pvalue[1])

#
# Define a useful text string describing the number of masked entries.  It is
# assumed that the input mask is a numpy mask where True means masked
#
def get_mask_info_string(mask):
    n_not_masked = np.sum(np.logical_not(mask))
    return '(#px=%i, used=%3.1f%%)' % (n_not_masked, 100 * n_not_masked/np.float64(mask.size))


def get_image_model_location(roots, b, dirs):
    image = os.path.join(ds.datalocationtools.save_location_calculator(roots, b)["image"])
    for d in dirs:
        image = os.path.join(image, d)
        if not(os.path.exists(image)):
            os.makedirs(image)
    return image
