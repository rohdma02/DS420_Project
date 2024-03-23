# Mirror file for importing pipeline into other files

from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer




# Step 2; based on housing_transformer_pipeline file

def load_creditcard_data():

    # filepath = Path("datasets/creditcard_2023.csv") 
    # if not filepath.is_file():
    #     Path("datasets").mkdir(parents=True, exist_ok=True)
    #     url = "https://github.com/ageron/data/raw/main/housing.tgz" 
    #     urllib.request.urlretrieve(url, filepath)
    #     with tarfile.open(tarball_path) as housing_tarball:
    #         housing_tarball.extractall(path="datasets") 

    return pd.read_csv("creditcard_2023.csv").drop(['id'], axis=1)


def create_pipeline():

    features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                'Amount']
    
    categorical_features = []


    # Create a transformer pipeline
    features_transformer = Pipeline(steps=[('imputer', SimpleImputer(
        strategy='median')), ('scaler', StandardScaler())])

    # Create a cat transformer pipeline
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(
        strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', features_transformer, features),
        ('cat', categorical_transformer, categorical_features)])

    # Create the final pipeline
    # add more steps later as we work on the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline


def prepare_creditcard_data():

    # Main

    creditcard_data = load_creditcard_data()
    

    X = creditcard_data.drop(['Class'], axis=1)
    y = creditcard_data['Class']

    
    # pipeline = create_pipeline()
    
    # pipeline.fit(X,y)

    return X, y