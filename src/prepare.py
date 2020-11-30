
from typing import Any, Dict
from sklearn import preprocessing
import pandas as pd

import sys
import os
import yaml
import pickle

data_filename = 'train.csv'

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)


pipe= sys.argv[1]
input = sys.argv[2]

if pipe == "test_prepare":
    data_filename = 'test.csv'

input_folder =  os.path.join('data', "train_prepare", 'encoder.pkl')

output_train = os.path.join('data', pipe, data_filename)
output_encoder = os.path.join('data', pipe, 'encoder.pkl')

params = yaml.safe_load(open('params.yaml'))[pipe]


def pre_process_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Some description.
    """
    col_to_encoder = params["col_to_encoder"]

    try:
        with open(input_folder, "rb") as input_file: 
            encoder = pickle.load(input_file)
        print("Loading Encoder")

    except:
        encoder= preprocessing.LabelEncoder()
        encoder.fit(data[col_to_encoder])

    data[col_to_encoder] = encoder.transform(data[col_to_encoder])

    # Has cabin boolean
    data.loc[:, 'has_cabin'] = 0
    data.loc[data['Cabin'].isna(), 'has_cabin'] = 1

    # Embarkment booleans
    for k in data.Embarked.unique():
        if type(k)==str:
            data['emb_' + k] = (data.Embarked==k)*1

    data = data.dropna()
    
    return data, encoder



os.makedirs(os.path.join('data', pipe), exist_ok=True)

data = pd.read_csv(input)
train, encoder = pre_process_data(data)
train.to_csv(output_train)
pickle.dump( encoder, open( output_encoder, "wb" ) )