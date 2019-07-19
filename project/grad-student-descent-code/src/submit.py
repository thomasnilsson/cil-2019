import pandas as pd
from datetime import datetime
import numpy as np
import os
import constants

def generate_submission_file(y_pred, submission_name="submission"):
    if len(y_pred) != 10000:
        print("Predicted vector must be exactly 10k elements long!")
        return
    # Replace zero entries with -1 (if any)
    y_pred[y_pred <= 0.5] = -1
    y_pred[y_pred > 0.5] = 1
    y_pred = y_pred.astype(int)

    # Make dataframe to export
    df = pd.read_csv(constants.SAMPLE_SUBMISSION_PATH)
    df.Prediction = y_pred
    
    # Timestamp for filename
    time =  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = submission_name + "_" + time + ".csv"
    path = os.path.join(constants.SUBMISSION_DIR, filename)
    
    # Export to csv
    df.to_csv(path, index=False)

    print("Submission %s saved!" % path)