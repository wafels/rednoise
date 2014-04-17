"""
Helpers for the paper
"""
from copy import deepcopy


#
#
#
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


#
# Line styles for all figures
#
class LineStyles:
    def __init__(self):
        self.color = 'b'
        self.linewidth = 1
        self.linestyle = "-"
        self.label = ""

s171 = LineStyles()
s171.color = 'b'

s_U68 = deepcopy(s171)
s_U68.label = '68%, upper'
s_U68.linestyle = "-"

s_L68 = deepcopy(s_U68)
s_L68.label = '68%, lower'
s_L68.color = 'r'

s_U95 = deepcopy(s171)
s_U95.label = '95%, upper'
s_U95.linestyle = "--"

s_L95 = deepcopy(s_U95)
s_L95.label = '95%, lower'
s_L95.color = 'r'


s193 = LineStyles()
s193.color = 'r'


s5min = LineStyles()
s5min.color = 'k'
s5min.linewidth = 3
s5min.linestyle = "-"
s5min.label = '5 mins.'

s3min = LineStyles()
s3min.color = 'k'
s3min.linewidth = 3
s3min.linestyle = "--"
s3min.label = '3 mins.'


#
# String defining the basics number of time series
#
def tsDetails(nx, ny, nt):
    return '[%i t.s., %i samples]' % (nx * ny, nt)

#
# Nicer names for printing
#
sunday_name = {'moss': 'moss',
               'qs': 'quiet Sun',
               'loopfootpoints': 'loop footpoints',
               'sunspot': 'sunspot'}

