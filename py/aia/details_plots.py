#
# Some magic numbers and parameters used in plots
#
import os
import re
import numpy as np
import matplotlib.cm as cm
import astropy.units as u
import details_study as ds


# Units that frequencies will be plotted in
fz = 'Hz'

# Which major color style for maps
map_color_style = 'w'

# General fontsize
fontsize = 20

#
plot_file_type = 'eps'

# Two different general map plot styles.  The first style has
# the sunspot outline in white, and the bad model fit pixels
# in white.  The second style has the opposite
map_plot_colors = {"w": {"sunspot": "white", "bad": "black"},
                   "k": {"sunspot": "black", "bad": "white"}}


class LinePlotStyle:
    """
    Storage class for line styles, to be used across multiple plots.
    """
    def __init__(self, color='k', label="", linestyle="-", linewidth=1,
                 position=1.0*u.s):
        self.color = color
        self.label = label
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.position = position
        self.frequency = 1.0 / self.position


class MapPlotStyle:
    """
    Storage class for map styles, to be used across multiple plots.
    """
    def __init__(self, cm=cm.plasma, bad='black', upper='blue', lower='cyan'):
        self.cm = cm
        self.bad = bad
        self.upper = upper
        self.lower = lower

#
# Line plot Styles
#
three_minutes = LinePlotStyle(color='k',
                              label='3 minutes',
                              linestyle=":",
                              linewidth=3,
                              position=180*u.s)


five_minutes = LinePlotStyle(color='k',
                             label='5 minutes',
                             linestyle="-",
                             linewidth=3,
                             position=300*u.s)


mean = LinePlotStyle(color='r', linewidth=3, linestyle='solid')
median = LinePlotStyle(color='r', linewidth=3, linestyle='dashed')
lo68 = LinePlotStyle(color='c', linewidth=3, linestyle='dashdot')
hi68 = LinePlotStyle(color='c', linewidth=3, linestyle='dotted')
mode = LinePlotStyle(color='r', linewidth=3, linestyle='dotted')
percentile0 = LinePlotStyle(color='r', linewidth=3, linestyle='dashdot')
percentile1 = LinePlotStyle(color='r', linewidth=3, linestyle='dashdot')


# Sunspot outline details
sunspot_outline = LinePlotStyle(color=map_plot_colors[map_color_style]["sunspot"], linewidth=3, linestyle='solid')

#
# Map Plot Styles
#
# Information criterion maps
information_criterion = MapPlotStyle(bad=map_plot_colors[map_color_style]["bad"],
                                     cm=cm.PiYG)

# Any power spectrum variable maps
amplitude_cm = cm.viridis
spectral_parameters = {'power law index': MapPlotStyle(cm=cm.Dark2_r),  # cm=cm.Dark_r
                       'ln(lognormal amplitude)': MapPlotStyle(cm=amplitude_cm),
                       'ln(constant)': MapPlotStyle(cm=amplitude_cm),
                       'ln(power law amplitude)': MapPlotStyle(cm=amplitude_cm),
                       'lognormal position': MapPlotStyle(cm=cm.Set2),
                       'ln(lognormal width)': MapPlotStyle(cm=cm.PiYG)}

# Histograms have the following type
hloc = (100,)  # 'scott', 'knuth', 'freedman')
histogram_1d_bins = 100

# Parameters which are frequencies
frequency_parameters = ['lognormal position']



#
# Useful strings for plots
#
# Reduced chi-squared
rchi2s = '$\chi^{2}_{r}$'
rchi2string = '$<$' + rchi2s + '$<$'

# Reduced chi-squared limit colors
rchi2limitcolor = ['w', 'w']


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
    number_pixel_string = "%i" % n_not_masked
    percent_used_string = '%3.1f%%' % (100 * n_not_masked/(1.0*mask.size))
    return number_pixel_string, percent_used_string, '#px=%s, used=%s' % (number_pixel_string, percent_used_string)


def get_image_model_location(roots, b, dirs):
    image = os.path.join(ds.datalocationtools.save_location_calculator(roots, b)["image"])
    for d in dirs:
        image = os.path.join(image, d)
        if not(os.path.exists(image)):
            os.makedirs(image)
    return image


#
# Concatenate a bunch of string elements
#
def concat_string(elements, sep='.'):
    output = elements[0]
    for element in elements[1:]:
        output += '%s%s' % (sep, element)
    return output


def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation.
    The algorithm plots out nicely formatted explicit numbers for values
    greater and less then 1.0."""
    if x >= 1.0:
        return '%i' % (x)
    else:
        n = 0
        while x * (10 ** n) <= 1:
            n = n + 1
        fmt = '%.' + str(n - 1) + 'f'
        return fmt % (x)


def clean_for_overleaf(s, rule='\W+', rep='_'):
    return re.sub(rule, rep, s)
