
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from handleYears import *

def linearReg_5points(series):
    
    point_2007 = series.iloc[-1]
    point_2006 = series.iloc[-2]
    point_2005 = series.iloc[-3]
    point_2004 = series.iloc[-4]
    point_2003 = series.iloc[-5]


    points = np.append(series.values[-5:-1], point_2007)
    years = np.array([float(y[:4]) for y in series.index[-5:]])


    z = np.isnan(points)  # Find the NaNs for later...
    zAny = np.any(np.isnan(points))  # Find the NaNs for later...



    # If only one of the last five years has a point, assume
    # that there will be no change
    if np.isnan(point_2006) and np.isnan(point_2005) and np.isnan(point_2004) and np.isnan(point_2003): 
        predictions = np.array([point_2007, point_2007])
    
    elif zAny == False:
        # This is the case that there are 5 good data points from the last 
        # five years

        pointX = years.reshape(-1,1)
        pointY = points.reshape(-1,1)
            
        lm = LinearRegression()
        lm.fit(pointX, pointY)
            
        pred_2008 = lm.predict(2008)
        pred_2012 = lm.predict(2012)

        if lm.score(pointX, pointY) < 0.2:
            print("BAD Score: ", series.name)

        predictions = np.array([pred_2008, pred_2012]).flatten()

    else:
        # The case where there are some NaN values in the last five
        # years data

        killem = []
        for i,j in enumerate(points):
            if np.isnan(j):
                killem.append(i)

        points = np.delete(points, killem)
        years = np.delete(years, killem)

        pointX = years.reshape(-1,1)
        pointY = points.reshape(-1,1)

        lm = LinearRegression()
        lm.fit(pointX, pointY)
        
        pred_2008 = lm.predict(2008)
        pred_2012 = lm.predict(2012)
        if lm.score(pointX, pointY) < 0.2:
            print("BAD Score: ", series.name)

        predictions = np.array([pred_2008, pred_2012]).flatten()
        

    # Set negative values to 0.0
    predictions[predictions < 0] = 0.01
    # Set values greater than 1.0 to 0.98
    predictions[predictions > 1.0] = 0.98

    ix = pd.Index(generate_year_list([2008, 2012]))
    return pd.Series(data=predictions, index=ix)
