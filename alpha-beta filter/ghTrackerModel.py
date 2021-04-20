import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class filterGeneric():

    def __init__(self):
        pass

    def g_h_filter(self, data, x0, dx, g, h, dt):
        """
        Performs g-h filter on 1 state variable with a fixed g and h.

        'data' contains the data to be filtered.
        'x0' is the initial value for our state variable
        'dx' is the initial change rate for our state variable
        'g' is the g-h's g scale factor
        'h' is the g-h's h scale factor
        'dt' is the length of the time step 
        """
        x = x0
        results = []
        lastDX = 0.
    
        for z in data:
            if np.isfinite(z):  # skip the NaNs
                # prediction step
                x_est = x + (dx*dt)
                dx = dx
                
                # update step
                residual = z - x_est
                dx = dx    + h * (residual) / dt
                x  = x_est + g * residual
                results.append(x)
                lastDX = dx
            else:
                results.append(np.nan)
                
        return np.array(results), lastDX

            
    def predict(self, data, dx):
        """
        Append points for one year and five years in the future
        to the array passed
        """
        predicted = np.array(data)


        # If the predicted value is above 1.0 (100%) or 0.0 (0%)
        # take the average of the last three measured values plus
        # 1.0 and use that instead  
        predictedNoNaN = np.array(data)  # Copy a version without NaNs
        predictedNoNaN = predictedNoNaN[~np.isnan(predictedNoNaN)]


        # Predict 1 year
        one = predicted[-1] + dx*1.0
        if one > 1.0:   # Some checks
            averageLast3one = predictedNoNaN[-1]
            if len(predictedNoNaN) > 2:
                averageLast3one = (predictedNoNaN[-3:].sum() + 1.0) / 4.
            elif len(predictedNoNaN) > 1:
                averageLast3one = (predictedNoNaN[-3:].sum() + 1.0) / 3.
            
            one = averageLast3one
            predictedNoNaNone = np.append(predictedNoNaN, one)
        elif one < 0.0:
            averageLast3one = predictedNoNaN[-1]
            if len(predictedNoNaN) > 2:
                averageLast3one = (predictedNoNaN[-3:].sum() + 0.0) / 4.
            elif len(predictedNoNaN) > 1:
                averageLast3one = (predictedNoNaN[-3:].sum() + 0.0) / 3.
            
            one = averageLast3one
            predictedNoNaNone = np.append(predictedNoNaN, one)
        predicted1 = np.append(predicted, one)

        # Predict 5 years
        five = predicted1[-1] + dx*5.0        
        if five > 1.0:
            averageLast3five = predictedNoNaN[-1]
            if len(predictedNoNaN) > 2:
                averageLast3five = (predictedNoNaN[-3:].sum() + 1.0) / 4.
            else:
                averageLast3five = (predictedNoNaN[-2:].sum() + 1.0) / 3.
            
            five = averageLast3five
        elif five < 0.0:
            averageLast3five = predictedNoNaN[-1]
            if len(predictedNoNaN) > 2:
                averageLast3five = (predictedNoNaN[-3:].sum() + 0.0) / 4.
            else:
                averageLast3five = (predictedNoNaN[-2:].sum() + 0.0) / 3.
            
            five = averageLast3five
        predicted2 = np.append(predicted1, five)


        # Return the extrapolated array (linear)        
        return predicted2
