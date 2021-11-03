from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from imblearn.over_sampling import SMOTE, RandomOverSampler
from model.cnn_lstm import CNN_LSTM
from miscellaneous.misc import Misc
from miscellaneous.error_estimation import error_estimation
from keras.models import load_model
import os
import joblib
import numpy as np
import pandas as pd


def run_smote(dataset_name=None, path_config=None, dataset_config=None, building_config=None, floor_config=None,
              positioning_config=None, algorithm=None):
    '''
    This function run all the machine learning models CNN-LSTM for building classification, floor classification,
    position prediction, and GAN
    :param dataset:
    :param dataset_config:
    :param building_config:
    :param floor_config:
    :param positioning_config:
    :param gan_full_config:
    :param data_augmentation:
    :return:
    '''

    misc = Misc()
    dataset_path = os.path.join(path_config['data_source'], dataset_name)
    if bool(dataset_config['train_dataset']):
        X_train, y_train = load(os.path.join(dataset_path, dataset_config['train_dataset']))

    if bool(dataset_config['test_dataset']):
        X_test, y_test = load(os.path.join(dataset_path, dataset_config['test_dataset']))

    if bool(dataset_config['validation_dataset']):
        X_valid, y_valid = load(os.path.join(dataset_path, dataset_config['validation_dataset']))
    else:
        X_valid = []
        y_valid = []
    # Data Normalization
    new_non_det_val = new_non_detected_value(X_train, X_test, X_valid)
    dr = DataRepresentation(x_train=X_train, x_test=X_test, x_valid=X_valid,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()

    oversample = SMOTE()
    X_train_overs, y_train_overs = oversample.fit_resample(X_train, y_train['FLOOR'])

    source_path = os.path.join(path_config['data_source'], dataset_name)
    saved_model_path = os.path.join(path_config['saved_model'], dataset_name, 'CNN-LSTM')

    # Load saved models
    position_model = load_model(saved_model_path + '/position.h5', compile=False)
    floor_model = load_model(saved_model_path + '/floor.h5', compile=False)
    building_model = load_model(saved_model_path + '/building.h5', compile=False)

    scaler_latitude = joblib.load(saved_model_path + '/lati_minmaxscaler.save')
    scaler_longitude = joblib.load(saved_model_path + '/long_minmaxscaler.save')
    scaler_altitude = joblib.load(saved_model_path + '/alti_minmaxscaler.save')

    # Data reshape

    if np.shape(X_train)[1] % 2 != 0:
        X_train_overs = np.append(X_train_overs, np.zeros((np.shape(X_train_overs)[0], 1)), axis=1)

    X_train_series_nd = X_train_overs.reshape((np.shape(X_train_overs)[0], np.shape(X_train_overs)[1], 1))
    subsequences = 2
    timesteps = X_train_series_nd.shape[1] // subsequences
    X_train_series_sub_nd = X_train_series_nd.reshape((X_train_series_nd.shape[0], subsequences, timesteps, 1))

    position = position_model.predict(X_train_series_sub_nd)
    floor = floor_model.predict(X_train_series_sub_nd)
    building = building_model.predict(X_train_series_sub_nd)

    predict_long = scaler_longitude.inverse_transform(position[:, 0].reshape(-1, 1))
    predict_lat = scaler_latitude.inverse_transform(position[:, 1].reshape(-1, 1))
    predict_alt = scaler_altitude.inverse_transform(position[:, 2].reshape(-1, 1))
    latitude = np.reshape(predict_lat[:], (1, len(predict_lat[:, 0])))
    longitude = np.reshape(predict_long[:], (1, len(predict_long[:, 0])))
    altitude = np.reshape(predict_alt[:], (1, len(predict_alt[:, 0])))

    floor = np.argmax(floor, axis=-1)
    building = np.argmax(building, axis=-1)

    new_data = list(zip(longitude[0], latitude[0], altitude[0], floor, building))

    new_y_train = pd.DataFrame(data=new_data, columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID'])

    # Training Model
    if np.size(X_valid) != 0:
        cnn_lstm = CNN_LSTM(X_train=X_train_overs, y_train=new_y_train.values, X_test=X_test, y_test=y_test.values,
                            X_valid=X_valid, y_valid=y_valid.values, dataset_config=dataset_config,
                            path_config=path_config, building_config=building_config, floor_config=floor_config,
                            position_config=positioning_config, algorithm=algorithm)
    else:
        cnn_lstm = CNN_LSTM(X_train=X_train_overs, y_train=new_y_train.values, X_test=X_test, y_test=y_test.values,
                            X_valid=X_valid, y_valid=y_valid, dataset_config=dataset_config,
                            path_config=path_config, building_config=building_config, floor_config=floor_config,
                            position_config=positioning_config, algorithm=algorithm)

    prediction = cnn_lstm.train()
    error_estimation(dataset_name, path_config, prediction, y_test, algorithm=algorithm)

    return True
