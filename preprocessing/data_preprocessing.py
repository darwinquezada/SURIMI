import pandas as pd
import seaborn as sns
import numpy as np
import scipy.io as sci
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def load(dataset):
    dataset = pd.read_csv(dataset)
    # Split the dataset into X data and y data
    X = dataset.iloc[:, :dataset.shape[1] - 5].values.copy().astype(np.float) # -9
    y = dataset.iloc[:, -5:].copy() # -9
    return X, y


def load_mat(dataset):
    db = {}
    dataset = sci.loadmat(dataset)
    db["trainingMacs"] = dataset["database"]['trainingMacs'].item()
    db["trainingLabels"] = dataset['database']['trainingLabels'].item()
    db["testMacs"] = dataset['database']['testMacs'].item()
    db["testLabels"] = dataset['database']['testLabels'].item()
    return db


def new_non_detected_value(X_train=[], X_test=[], X_valid=[]):
    if np.size(X_valid) != 0:
        new_non_dect_value = np.min(np.min(np.concatenate((X_train, X_test, X_valid)))) - 1
    else:
        new_non_dect_value = np.min(np.min(np.concatenate((X_train, X_test)))) - 1
    return new_non_dect_value


def data_reshape_st(X_train, X_test, X_valid):
    '''
    Resampling data [sample, timestep]
    :param X_train:
    :param X_test:
    :param X_valid:
    :return:
    '''
    if np.shape(X_train)[1] % 2 != 0:
        X_train = np.append(X_train, np.zeros((np.shape(X_train)[0], 1)), axis=1)
    if np.shape(X_test)[1] % 2 != 0:
        X_test = np.append(X_test, np.zeros((np.shape(X_test)[0], 1)), axis=1)

    X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    if np.size(X_valid) != 0:
        if np.shape(X_valid)[1] % 2 != 0:
            X_valid = np.append(X_valid, np.zeros((np.shape(X_valid)[0], 1)), axis=1)
        X_valid_series = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    else:
        X_valid_series = X_valid
    return X_train_series, X_test_series, X_valid_series


def data_reshape_stf(X_train, X_test, X_valid):
    '''
    Resampling from [sample, timestep] to [sample, timestep, feature]
    :param X_train:
    :param X_test:
    :param X_valid:
    :return:
    '''
    X_train_series, X_test_series, X_valid_series = data_reshape_st(X_train, X_test, X_valid)
    subsequences = 2
    timesteps = X_train_series.shape[1] // subsequences
    X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
    X_test_series_sub = X_test_series.reshape((X_test_series.shape[0], subsequences, timesteps, 1))
    if np.size(X_valid_series) != 0:
        X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))
    else:
        X_valid_series_sub = X_valid_series
    return X_train_series_sub, X_test_series_sub, X_valid_series_sub


class Normalize():
    def __init__(self):
        self.scaler = MinMaxScaler()

    def normalize_y(self, latitude, longitude):
        latitude = self.scaler.fit_transform(latitude.reshape(-1, 1))
        longitude = self.scaler.fit_transform(longitude.reshape(-1, 1))
        return latitude, longitude

    def reverse_normalize_y(self, latitude, longitude):
        latitude = self.scaler.inverse_transform(latitude.reshape(-1, 1))
        longitude = self.scaler.inverse_transform(longitude.reshape(-1, 1))
        return latitude, longitude
