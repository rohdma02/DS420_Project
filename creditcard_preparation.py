# Mirror file for importing pipeline into other files

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



def create_pipeline(features, categorical_features):

    # Create a trasnformer pipeline
    features_transformer = Pipeline(steps=[('imputer', SimpleImputer(
        strategy='median')), ('scaler', StandardScaler())])

    # Create a cat transformer pipeline
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(
        strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[(
        'num', features_transformer, features), ('cat', categorical_transformer, categorical_features)])

    # Create the final pipeline
    # add more steps later as we work on the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline


def main():

    # Demo

    csv_file = "creditcard_2023.csv"

    #Read data from csv into DataFrame
    df = pd.read_csv(csv_file)

    X = df.drop(['id', 'Class'], axis=1)
    y = df['Class']

    features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                'Amount']
    categorical_features = []
    pipeline = create_pipeline(features, categorical_features)
    pipeline.fit(X,y)