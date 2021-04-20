
import numpy as np
import pandas as pd
import statsmodels.api as sm


from linearRegressionModel import *
from handleYears import *
from utility import *


training_data = pd.read_csv("../data/TrainingSet.csv", index_col=0)
submission_labels = pd.read_csv("../data/SubmissionRows.csv", index_col=0)


prediction_rows = training_data.loc[submission_labels.index]
prediction_rows = prediction_rows[generate_year_list(1972, 2007)]  # Gets rid of the nonsense columns


# Apply model and make some predictions
predictions = prediction_rows.apply(linearReg_5points, axis=1)
    
write_submission_file(predictions, "Attempt2.csv")
