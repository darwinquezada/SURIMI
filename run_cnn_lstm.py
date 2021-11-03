from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from model.cnn_lstm import CNN_LSTM
from miscellaneous.misc import Misc
from miscellaneous.error_estimation import error_estimation
import os
import numpy as np


def run_cnn_lstm(dataset_name=None, path_config=None, dataset_config=None, building_config=None, floor_config=None,
                 positioning_config=None, algorithm=None):
    '''
    This function run all the machine learning models CNN-LSTM for building classification, floor classification,
    position prediction, and GAN
    :param algorithm:
    :param path_config:
    :param dataset_name:
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

    # Training Model
    if np.size(X_valid) != 0:
        cnn_lstm = CNN_LSTM(X_train=X_train, y_train=y_train.values, X_test=X_test, y_test=y_test.values,
                            X_valid=X_valid, y_valid=y_valid.values, dataset_config=dataset_config,
                            path_config=path_config, building_config=building_config, floor_config=floor_config,
                            position_config=positioning_config, algorithm=algorithm)
    else:
        cnn_lstm = CNN_LSTM(X_train=X_train, y_train=y_train.values, X_test=X_test, y_test=y_test.values,
                            X_valid=X_valid, y_valid=y_valid, dataset_config=dataset_config,
                            path_config=path_config, building_config=building_config, floor_config=floor_config,
                            position_config=positioning_config, algorithm=algorithm)

    prediction = cnn_lstm.train()
    error_estimation(dataset_name, path_config, prediction, y_test, algorithm=algorithm)

    return True
