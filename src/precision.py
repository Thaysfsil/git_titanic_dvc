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

pipe = "model_precision"

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)


input = sys.argv[1]
model_path = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))[pipe]


def report_accuracy(model: np.ndarray, x_train: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """

    log = logging.getLogger(__name__)
    features = params["features"] 
    target = params["target"]

    y_predicted = model.predict(x_train[features])
    precision = precision_score(x_train[target], y_predicted)
    log.info(f"Model precision on train set: {np.mean(precision)},  STD: { np.std(precision)}")




os.makedirs(os.path.join('data', pipe), exist_ok=True)

data = pd.read_csv(input)
with open(model_path, "rb") as input_file: 
    model = pickle.load(input_file)
report_accuracy(model, data)
