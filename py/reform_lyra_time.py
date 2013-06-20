import re
from sunpy.time import *

def reform_lyra_time(input_time):
    q=re.split(' ',input_time)
    if len(q) == 2:
        date=q[0]
    else:
        date=q[1]
    datestr=re.split('-',date)
    newdate=datestr[2]+ '-' + datestr[1] + '-' + datestr[0]
    if len(q) == 2:
        time=re.split('\.',q[1])
    else:
        time=re.split('\.',q[2])
    newdatestring=newdate + ' ' + time[0]
    return newdatestring
    
