# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from ucimlrepo import fetch_ucirepo
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def downloading_data():
    ''' Downloading data'''
    # fetch dataset
    bank_marketing = fetch_ucirepo(id=222)
    # data (as pandas dataframes)
    X = bank_marketing.data.features # shape = (45211, 16)
    y = bank_marketing.data.targets # shape = (45211, 1)

    # metadata
    print(bank_marketing.metadata)
    # variable information
    print(bank_marketing.variables)

    return X, y

# First download data
X, y = downloading_data()


def data_exploration(X, y):
    """
    Using the data provided calculate: n_records --> number of samples, n_subscriber --> number of  the client subscribed a term deposit,
    subscriber_percent --> percentage of the client subscribed a term deposit.
    Input: data (pd.DataFrame)
    Output: (n_records, n_subscriber, subscriber_percent) -> Tuple of integers
    """
    # TODO : write your code here

    return n_records, n_subscriber, subscriber_percent

def feature_encoding(X):
    """
    One-hot encode the 'features'.
    Input: X: features (pd.DataFrame) with shape = (45211, 16)
    Output: X: features_encoded (pd.DataFrame) with shape = (45211, 16)
    """
    non_numerical_columns_names = X.select_dtypes(exclude=['number']).columns
    # TODO : write encoding here

    return X


def encode_label(y):
    """
    Encode the 'labels' data to numerical values.
    Input: y: labels (pd.DataFrame) with shape = (45211, 1)
    Output: y: labels_int (pd.DataFrame) with shape = (45211, 1)
    """
    # TODO : write encoding here

    return y



def data_preprocessing():
    # First download data
    X, y = downloading_data()
    # convert categorical to numerical
    X = feature_encoding(X)
    y = encode_label(y)

    return X, y