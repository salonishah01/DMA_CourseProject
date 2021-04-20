
import pandas as pd
import numpy as np

def generate_year_list(start, stop=None):
    """ 
    make a list of column names for specific years
    in the format they appear in the data frame start/stop inclusive
    """
    
    if isinstance(start, list):
        data_range = start
    elif stop:
        data_range = range(start, stop+1)
    else:
        data_range = [start]
    
    yrs = []
    
    for yr in data_range:
        yrs.append("{0} [YR{0}]".format(yr))
        
    return yrs



def floatYr(start, stop):
    """
    return a list of years as floats
    """
    
    yr = [float(x[0:4]) for x in generate_year_list(start,stop)]

    return yr
