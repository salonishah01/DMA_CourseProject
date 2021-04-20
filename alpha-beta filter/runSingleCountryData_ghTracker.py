
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ghTrackerModel import *
from handleYears import *
from utility import *



training_data = pd.read_csv("../data/TrainingSet.csv", index_col=0)
submission_labels = pd.read_csv("../data/SubmissionRows.csv", index_col=0)


# Pick the country to investigate. The script will plot the time series for this 
# country for the values that are requested in the submission file.  This is 
# always less than 6 per country


#lookingFOR = "Sweden"
#lookingFOR = "South Africa"
#lookingFOR = "Costa Rica"
#lookingFOR = "Portugal"
#lookingFOR = "Cuba"
#lookingFOR = "Georgia"
#lookingFOR = "Bolivia"
lookingFOR = "Antigua and Barbuda"



singleC_data2 = training_data.loc[submission_labels.index]
singleC_data = singleC_data2[singleC_data2["Country Name"] == lookingFOR]

singleC_dataN = singleC_data.apply(pd.to_numeric, errors='coerce')  # make the string nan into np.nan
yr = floatYr(1972, 2007)
yrPredict = list(yr)
yrPredict.append(2008.)
yrPredict.append(2012.)


# Find the indices for this country to loop over
predictInd = singleC_data.index



fig, ax =  plt.subplots(3, 2)
ct1 = 0
ct2 = 0

for ii in predictInd:

    singleC_dataSm = singleC_dataN.loc[ii]
    singleC_valuesSmall = singleC_dataSm[generate_year_list(1972, 2007)].values  # return numpy object


    # Start value for the slope/gain --- highest/lowest divided by the number of non-NaN values in the 
    # series.  In other words, the average gain.
    gain = (singleC_dataSm[:-3].max() - singleC_dataSm[:-3].min()) / float(np.count_nonzero(~np.isnan(singleC_valuesSmall)))

    # Find the start values for the filter, first non-NaN cell
    indices = np.where(~np.isnan(singleC_valuesSmall))


    # Define the filter, set it rolling
    f = filterGeneric()
    data, lastDX = f.g_h_filter(data=singleC_valuesSmall, x0=singleC_valuesSmall[indices[0][0]], dx=gain, g=7./10, h=2./3, dt=1.)


    # Predict 2008, 2012
    predicted = f.predict(data, lastDX)


    # If the three values prior to 2008/2012 were all the same,
    # overwrite and just use that value -- assume constant
    dataNoNaN = np.array(data)  # Copy a version without NaNs
    dataNoNaN = dataNoNaN[~np.isnan(dataNoNaN)]
    if (len(dataNoNaN) > 2):
        frac23 = (dataNoNaN[-3]-dataNoNaN[-2])/dataNoNaN[-3]
        frac12 = (dataNoNaN[-2]-dataNoNaN[-1])/dataNoNaN[-2]
        if (np.absolute(frac23)<0.01) & (np.absolute(frac12)<0.01):
            predicted[-1] = predicted[-3]
            predicted[-2] = predicted[-3]
    elif (len(dataNoNaN) == 2):
        frac12 = (dataNoNaN[-2]-dataNoNaN[-1])/dataNoNaN[-2]
        if (np.absolute(frac12)<0.01):
            predicted[-1] = predicted[-3]
            predicted[-2] = predicted[-3]
    elif (len(dataNoNaN) == 1):
        predicted[-1] = predicted[-3]
        predicted[-2] = predicted[-3]



    labelName1 = "data - " + str(ii)
    labelName2 = "pred - " + str(ii)

    ax[ct1, ct2].plot(yr,singleC_valuesSmall,marker="o",c="Black", label=labelName2)
    ax[ct1, ct2].plot(yrPredict,predicted,marker="o",c="Blue", ls='--')
    ax[ct1, ct2].plot(yr,data,marker="o",c="Blue", label=labelName1)
    

    if ct1==0 & ct2==0:
        ct1 = 1
    elif ct1==1 and ct2==0:
        ct1 = 2
        ct2 = 0
    elif ct1==2 and ct2==0:
        ct1 = 0
        ct2 = 1
    elif ct1==0 and ct2==1:
        ct1 = 1
        ct2 = 1
    elif ct1==1 and ct2==1:
        ct1 = 2
        ct2 = 1


saveName = "cPredictors" + lookingFOR + ".png"
plt.savefig(saveName, format='png', dpi=300)

write_submission_file(predictions, "Attempt.csv")
