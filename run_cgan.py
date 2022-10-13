from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from model.cnn_lstm import CNN_LSTM
from miscellaneous.misc import Misc
from miscellaneous.error_estimation import error_estimation
from miscellaneous.fingerprints_generation import generation
from model.cgan_full_db import train_imbalance_classes
from model.cgan_Building import train_imbalance_classes_building
from model.cgan_Floor import train_imbalance_classes_floor
import os
import numpy as np
import pandas as pd


def run_cgan(dataset_name=None, path_config=None, dataset_config=None, building_config=None, floor_config=None,
             positioning_config=None, gan_general_config=None, data_augmentation=None, algorithm=None,
             method=None):
    """

    Parameters
    ----------
    dataset_name : Dataset name
    path_config : General paths set in the config file
    dataset_config : Dataset configuration
    building_config : Hyperparameters for the building model
    floor_config : Hyperparameters for the floor model
    positioning_config : Hyperparameters for the positioning model
    gan_general_config : Hyperparameters for the cGAN model
    data_augmentation : Data augmentatio parameters set in the config file
    algorithm: Algorithm to be used, for now CGAN
    method: Method to be used to train the GAN model (FLOOR, BUILDING, FULL_DB)

    Returns
    -------

    """

    misc = Misc()
    # Load datasets
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

    # Generate new fingerprints with cGAN
    # GAN Model
    for model in gan_general_config['gan']['model']:
        if model['model'] == 'discriminator':
            discriminator_config = model
        elif model['model'] == 'generator':
            generator_config = model
        elif model['model'] == 'gan':
            gan_config = model
        else:
            misc.log_msg('INFO', model['model'] + ' not defined.')

    # Move all the files
    source_folder = os.path.join(path_config['saved_model'], dataset_name, algorithm)
    string_dist = [str(dist) for dist in data_augmentation['distance_rsamples']]
    conf_augment = 'epochs_' + str(gan_general_config['epochs']) + '_bs_' + str(gan_general_config['batch_size']) + \
                   '_dist_(' + ','.join(string_dist) + ')_iter_' + str(data_augmentation['iterations'])
    destination_folder = os.path.join(source_folder, method, conf_augment)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Train GAN architecture
    if gan_general_config['train'].upper() == 'TRUE':
        # Train the model by floor
        if method == 'FLOOR':
            train_imbalance_classes_floor(X_train, y_train, dataset_config=dataset_config,
                                          discriminator_config=discriminator_config,
                                          gan_general_config=gan_general_config, gan_config=gan_config,
                                          path_config=path_config,
                                          algorithm=algorithm, method=method)
        # Train the model by building
        elif method == 'BUILDING':
            train_imbalance_classes_building(X_train, y_train, dataset_config=dataset_config,
                                             discriminator_config=discriminator_config,
                                             gan_general_config=gan_general_config, gan_config=gan_config,
                                             path_config=path_config,
                                             algorithm=algorithm, method=method)
        else:
            # Train the model using the full dataset
            method = 'FULL-DB'
            train_imbalance_classes(X_train, y_train, dataset_config=dataset_config,
                                    discriminator_config=discriminator_config,
                                    gan_general_config=gan_general_config, gan_config=gan_config,
                                    path_config=path_config,
                                    algorithm=algorithm, method=method)

    # Generate new fingerprints ("synthetic fingerprints")
    if gan_general_config['generate_fp'].upper() == 'TRUE':
        X_train_new, y_train_new = generation(dataset_name=dataset_name, dataset_config=dataset_config,
                                              path_config=path_config, gan_general_config=gan_general_config,
                                              data_augmentation=data_augmentation, algorithm=algorithm,
                                              method=method)

    else:
        string_dist = [str(dist) for dist in data_augmentation['distance_rsamples']]
        conf_augment = 'epochs_' + str(gan_general_config['epochs']) + '_bs_' + str(gan_general_config['batch_size']) +\
                       '_dist_(' + ','.join(string_dist) + ')_iter_' + str(data_augmentation['iterations'])
        path_file = os.path.join(path_config['saved_model'], dataset_name, algorithm, method, conf_augment)

        if path_file:
            X_train_new = pd.read_csv(path_file + '/TrainingData_x_augmented.csv')
            y_train_new = pd.read_csv(path_file + '/TrainingData_y_augmented.csv')
        else:
            print(misc.log_msg('ERROR', 'Error. Please, train the model first.'))
            exit(-1)

    # Concatenate synthetic fingerprints and original fingerprints in the training set
    X_train = np.concatenate((X_train, X_train_new.iloc[:, 0:np.shape(X_train)[1]].values), axis=0)
    y_train = y_train.append(y_train_new)


    prediction_path = os.path.join(path_config['results'], dataset_config['name'], algorithm, method, conf_augment)

    # Training the CNN-LSTM Model
    if building_config['train'].upper() == 'TRUE' or floor_config['train'].upper() == 'TRUE' or \
            positioning_config['train'].upper() == 'TRUE':

        if np.size(X_valid) != 0:
            cnn_lstm = CNN_LSTM(X_train=X_train, y_train=y_train.values, X_test=X_test, y_test=y_test.values,
                                X_valid=X_valid, y_valid=y_valid.values, dataset_config=dataset_config,
                                path_config=path_config, building_config=building_config, floor_config=floor_config,
                                position_config=positioning_config, algorithm=algorithm,
                                gan_general_config=gan_general_config, data_augmentation=data_augmentation,
                                method=method)
        else:
            cnn_lstm = CNN_LSTM(X_train=X_train, y_train=y_train.values, X_test=X_test, y_test=y_test.values,
                                X_valid=X_valid, y_valid=y_valid, dataset_config=dataset_config,
                                path_config=path_config, building_config=building_config, floor_config=floor_config,
                                position_config=positioning_config, algorithm=algorithm,
                                gan_general_config=gan_general_config, data_augmentation=data_augmentation,
                                method=method)

        prediction = cnn_lstm.train()
        # Save prediction
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        prediction.to_csv(prediction_path+'/prediction.csv', header=True, index=False)
    else:
        if os.path.isfile(prediction_path+'/prediction.csv'):
            prediction = pd.read_csv(prediction_path+'/prediction.csv')
        else:
            print(misc.log_msg('ERROR',
                               'Error. Please, train the positioning model first (position, floor and building).'))
            exit(-1)
    # Report results
    error_estimation(database_name=dataset_name, path_config=path_config, prediction=prediction, test=y_test,
                     algorithm=algorithm, conf_augment=conf_augment, method=method)

    return True
