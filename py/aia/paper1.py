"""
Helpers for the paper
"""


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


s171 = LineStyles()
s171.color = 'b'

s193 = LineStyles()
s193.color = 'r'

s5min = LineStyles()
s5min.color = 'k'
s5min.linewidth = 3
s5min.linestyle = "-"

s3min = LineStyles()
s3min.color = 'k'
s3min.linewidth = 3
s3min.linestyle = "--"


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

