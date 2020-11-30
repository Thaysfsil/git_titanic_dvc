
from typing import Any, Dict
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import pickle
import yaml
import sys
import os


pipe = "model_train"
if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)



input = sys.argv[1]

output_model = os.path.join('data', pipe, "model.pkl")

params = yaml.safe_load(open('params.yaml'))[pipe]



def train_model( train: pd.DataFrame) -> Any:
    """Some desdcription """

    features = params["features"] 
    target = params["target"]
    model_param = params["model_param"]

    #Árvore de decisão 
    dtree=DecisionTreeClassifier(**model_param)

    # Here we train our final model against all of our validation data. 
    dtree.fit(train[features], train[target])

    return dtree



os.makedirs(os.path.join('data', pipe), exist_ok=True)

data = pd.read_csv(input)
model = train_model(data)
pickle.dump( model, open( output_model, "wb" ) )

