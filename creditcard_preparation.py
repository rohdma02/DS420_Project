# Preparing Data

# Authors: Cody, Mateus, and Mughees
# Step 2: Data Preparation


# Manually maintained mirror file for importing pipeline into other files

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


import ssl

# may be needed to retrieve data
ssl._create_default_https_context = ssl._create_unverified_context




# Step 2; based on housing_transformer_pipeline file

def load_creditcard_data():
    
    base_file = "https://github.com/rohdma02/DS420_Project/blob/main/data/creditcard_2023_1.csv?raw=True"
    
    additional_files = ["https://github.com/rohdma02/DS420_Project/blob/main/data/creditcard_2023_2.csv?raw=True",
             "https://github.com/rohdma02/DS420_Project/blob/main/data/creditcard_2023_3.csv?raw=True",
             "https://github.com/rohdma02/DS420_Project/blob/main/data/creditcard_2023_4.csv?raw=True"]
    
    df = pd.read_csv(base_file)
    
    for url in additional_files:
        
        more_rows = pd.read_csv(url)
        
        df = pd.concat([df, more_rows])
        
        
    return df.drop(['id'], axis=1)



def create_creditcard_pipeline():

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




def split_creditcard_data(data, ratios):

    # Shuffle data

    randomized_data = data.sample(frac = 1, random_state=1)


    if 'id' in randomized_data.columns:

        randomized_data = randomized_data.drop(['id'], axis=1)


    X = randomized_data.drop(['Class'], axis=1)
    y = randomized_data['Class']

    
    # Get ratios from tuple
    dev_ratio = ratios[0]
    test_ratio = ratios[1]
    
    # Determine size of sets using given ratios
    devset_size = int(dev_ratio * X.shape[0])
    testset_size = int(test_ratio * X.shape[0])
    
    
    # Take data points up to number needed for devset as training set
    X_train = X[:-(devset_size+testset_size)]
    y_train = y[:-(devset_size+testset_size)]
    
    
    # Take devset_size data points before testset_size data points for dev set
    X_dev = X[-(devset_size+testset_size):-testset_size]
    y_dev = y[-(devset_size+testset_size):-testset_size]
    
    
    #Take last testset_size data points as test set
    X_test = X[-testset_size:]
    y_test = y[-testset_size:]
    

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def prepare_creditcard_data(ratios):

    creditcard_data = load_creditcard_data()
    
    return split_creditcard_data(creditcard_data, ratios)