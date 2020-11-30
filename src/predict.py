


import logging
from typing import Any, Dict
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import sys
import os
import yaml
import pickle


pipe = "model_predict"

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)


input = sys.argv[1]
model_path = sys.argv[2]

output_prediction = os.path.join('data', pipe, "df_prediction.csv")

params = yaml.safe_load(open('params.yaml'))[pipe]



def predict(model: Any, test_x: pd.DataFrame) -> pd.DataFrame:
    """Node for making predictions given a pre-trained model and a test set.
    """
    features = params["features"] 
    test_probabilities = model.predict_proba(test_x[features])[:,1]
    test_x['survival_likelihood'] = test_probabilities

    return test_x

os.makedirs(os.path.join('data', pipe), exist_ok=True)

data = pd.read_csv(input)
with open(model_path, "rb") as input_file: 
    model = pickle.load(input_file)
prediction = predict(model, data)
prediction.to_csv(output_prediction)


