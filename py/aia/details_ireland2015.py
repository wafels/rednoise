#
# Some results from Ireland et al 2015, ApJ, 798, 1
#
import os
import pandas as pd

# Name of the data file
filename = 'ireland2015_table1.csv'

# Directory
directory = os.path.expanduser('~/ts/rednoise/dat/')

# Full file path
filepath = os.path.join(directory, filename)

# Load the data in to a dataframe
df = pd.DataFrame.from_csv(filepath)

#
label = 'Ireland et al 2015'

"""
#
# Example to get a value out
#
z=df.loc['moss']

power_law_index = z[ z['waveband'] == 171]['n']
"""