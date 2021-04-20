import numpy as np
import pandas as pd
import collections as col
import matplotlib.pyplot as plt


from ghTrackerModel import *
from handleYears import *
from utility import *



training_data = pd.read_csv("../data/TrainingSet.csv", index_col=0)
submission_labels = pd.read_csv("../data/SubmissionRows.csv", index_col=0)


# The rows/values to be predicted
prediction_rows = training_data.loc[submission_labels.index]


# Output submission file format
forCSVstart = np.array(prediction_rows.index).reshape(len(prediction_rows),1)
forCSV = np.concatenate((forCSVstart, np.zeros(shape=(len(forCSVstart),2))), 1)


# Generate the list of countries to loop over...
cc2 = col.Counter(prediction_rows['Country Name'])


for num, xx in enumerate(cc2):

    # Standardize country name
    yy = xx.replace(" ", "_")
    zz = yy.replace(",", "_")
    ff = zz.replace(".", "_")
    lookingFOR = ff.replace("'", "_")
    print("Looking at ", lookingFOR, num)


    singleC_data2 = training_data.loc[submission_labels.index]
    singleC_data = singleC_data2[singleC_data2["Country Name"] == xx]


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
        data, lastDX = f.g_h_filter(data=singleC_valuesSmall, x0=singleC_valuesSmall[indices[0][0]], dx=gain, g=7.4/10, h=2.5/3, dt=1.)


        # Predict 2008, 2012
        predicted = f.predict(data, lastDX)


        # If the three values prior to 2008/2012 were all the same,
        # overwrite and just use that value -- assume constant
        dataNoNaN = np.array(singleC_valuesSmall)  # Copy a version without NaNs
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


        # If the predicted value is >15% larger than the previous version, half it...
        if (np.absolute(predicted[-1] - dataNoNaN[-1]) > 0.15):
            print ("LESS THAN 15%, -1", predicted[-1], dataNoNaN[-1])
            if (predicted[-1] - dataNoNaN[-1]) > 0.:                
                predicted[-1] = dataNoNaN[-1] + np.absolute(predicted[-1] - dataNoNaN[-1]) / 2.0
                print ("Updated predicted:", predicted)
            else:
                predicted[-1] = dataNoNaN[-1] - np.absolute(predicted[-1] - dataNoNaN[-1]) / 2.0
                "Updated predicted:", predicted
        if (np.absolute(predicted[-2] - dataNoNaN[-1]) > 0.15):
            print ("LESS THAN 15%, -2", predicted[-1], dataNoNaN[-1])
            if (predicted[-2] - dataNoNaN[-1]) > 0.:                
                predicted[-2] = dataNoNaN[-1] + np.absolute(predicted[-2] - dataNoNaN[-1]) / 2.0
                "Updated predicted:", predicted
            else:
                predicted[-2] = dataNoNaN[-1] - np.absolute(predicted[-2] - dataNoNaN[-1]) / 2.0
                "Updated predicted:", predicted


        # Push any that may be >1.0 or <0.0 back into the fold!
        if predicted[-1] > 1.0:
            predicted[-1] = 0.99
        if predicted[-2] > 1.0:
            predicted[-2] = 0.99
        if predicted[-1] < 0.0:
            predicted[-1] = 0.1
        if predicted[-2] < 0.0:
            predicted[-2] = 0.1            

                    

        # Clunky --- but... find the indice to save the predicted values...
        indice = np.where((forCSV==ii)[:,0])[0][0]
        forCSV[indice][1] = predicted[-2]
        forCSV[indice][2] = predicted[-1]
        print ("FILLING...\n", forCSV, "\n", forCSV[indice])

        
        labelName1 = "data - " + str(ii)
        labelName2 = "pred - " + str(ii)
        

        # Plot
        ax[ct1, ct2].plot(yr,singleC_valuesSmall,marker="o",c="Black", label=labelName2)
        ax[ct1, ct2].plot(yrPredict,predicted,marker="o",c="Blue", ls='--')
        ax[ct1, ct2].plot(yr,data,marker="o",c="Blue", label=labelName1)
        ax[ct1, ct2].legend(loc="upper left", fontsize="x-small")

        
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
            

    saveName = "plots/cPredictors" + lookingFOR + ".png"
    plt.savefig(saveName, format='png', dpi=300)
    plt.close('all')

write_submission_file(forCSV, "AttemptX_ghfilter.csv")
