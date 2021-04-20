

import os
import pandas as pd
import numpy as np
from handleYears import *


def write_submission_file(preds, filename):
    """
    Utility function to output submission file
    """


    # load the submission labels
    file_format = pd.read_csv(os.path.join("data", "SubmissionRows.csv"), index_col=0)
    expected_row_count = file_format.shape[0]

    if isinstance(preds, pd.DataFrame):
        # check indices
        assert(preds.index == file_format.index).all(), \
            "DataFrame: Prediction indices must match submission format."
        
        # check columns
        assert (preds.columns == file_format.columns).all(), \
            "DataFrame: Column names must match submission format."
        
        final_predictions = preds
        
    elif isinstance(preds, np.ndarray):
        rows, cols = preds.shape
        
        if cols == 3:
            assert (preds[:,0] == file_format.index.values).all(), \
                "Numpy Array: First column must be indices."
            
            # now we know the indices are cool, ditch them
            preds = preds[:,1:]
        
        assert rows == expected_row_count, \
            "Numpy Array: The predictions must have the right number of rows."
        
        # put the predictions into the dataframe
        final_predictions = file_format.copy()
        final_predictions[generate_year_list([2008, 2012])] = preds
            
    elif isinstance(preds, list):
        assert len(preds) == 2, \
            "list: Predictions must be a list containing two lists"
        assert len(preds[0]) == expected_row_count, \
            "list: There must be the right number of predictions in the first list."
        assert len(preds[1]) == expected_row_count, \
            "list: There must be the right number of predictions in the second list."
    
        # write the predictions
        final_predictions = file_format.copy()
        final_predictions[generate_year_list(2008)] = np.array(preds[0], dtype=np.float64).reshape(-1, 1)
        final_predictions[generate_year_list(2012)] = np.array(preds[1], dtype=np.float64).reshape(-1, 1)
        
    elif isinstance(preds, dict):
        assert preds.keys() == generate_year_list([2008, 2012]), \
            "dict: keys must be properly formatted"
        assert len(preds[generate_year_list(2008)[0]]) == expected_row_count, \
            "dict: length of value for 2008 must match the number of predictions"
        assert len(preds[generate_year_list(2012)[0]]) == expected_row_count, \
            "dict: length of value for 2012 must match the number of predictions"
        
        # create dataframe from dictionary
        final_predictions = pd.DataFrame(preds, index=file_format.index)

    final_predictions.to_csv(filename)


