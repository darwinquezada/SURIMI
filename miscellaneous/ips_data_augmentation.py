from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from collections import Counter
from miscellaneous.misc import Misc
from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from model.cnn_lstm import CNN_LSTM
import numpy as np
import pandas as pd
import glob
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def data_augmentation(dataset_name=None, dataset_config=None, path_files=None, gan_general_config=None):
    source_path = os.path.join(path_files['data_source'], dataset_name)
    saved_model_path = os.path.join(path_files['saved_model'], dataset_name)

    # Load saved models
    longitude_model = load_model(saved_model_path + '/pos-long.h5', compile=False)
    latitude_model = load_model(saved_model_path + '/pos-lati.h5', compile=False)
    floor_model = load_model(saved_model_path + '/floor.h5', compile=False)
    building_model = load_model(saved_model_path + '/building.h5', compile=False)

    scaler_latitude = joblib.load(saved_model_path + '/lati_minmaxscaler.save')
    scaler_longitude = joblib.load(saved_model_path + '/long_minmaxscaler.save')
    # encoder_floor = joblib.load(saved_model_path + '/floor_onehotencoder.save')
    # encoder_building = joblib.load(saved_model_path + '/building_onehotencoder.save')

    misc = Misc()
    dataset_path = os.path.join(path_files['data_source'], dataset_name)
    if bool(dataset_config['train_dataset']):
        X_train, y_train = load(os.path.join(dataset_path, dataset_config['train_dataset']))

    if bool(dataset_config['test_dataset']):
        X_test, y_test = load(os.path.join(dataset_path, dataset_config['test_dataset']))

    if bool(dataset_config['validation_dataset']):
        X_valid, y_valid = load(os.path.join(dataset_path, dataset_config['validation_dataset']))
    else:
        X_valid = []

    # Data Normalization
    new_non_det_val = new_non_detected_value(X_train, X_test, X_valid)
    dr = DataRepresentation(x_train=X_train, x_test=X_test, x_valid=X_valid,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()

    for b_idx, bld in enumerate(np.unique(y_test['BUILDINGID'])):
        # Data selection
        floor_to_analyse = []
        os.chdir(saved_model_path)
        for file in glob.glob("cgan_generator_b_" + str(bld) + "_f_*_i_*.h5"):
            split_file = file.split("_")
            floor_to_analyse.append(split_file[5])

        unique_floors = np.unique(floor_to_analyse)

        # Get the class with more samples
        classes_val = Counter(y_train.values[:, 2])
        class_val_sort = classes_val.most_common()
        max_class_val = class_val_sort[0][1]

        df_full_augmented_y_train = pd.DataFrame()
        df_full_augmented_x_train = pd.DataFrame()

        for id, flr in enumerate(unique_floors):
            # Select samples
            index_samples = y_train[y_train['FLOOR'] == int(flr)].index
            X_train_selected = X_train[index_samples, :]
            y_train_selected = y_train[y_train['FLOOR'] == int(flr)]

            df_new_fake_labels = pd.DataFrame()
            df_new_fake_fingerprints = pd.DataFrame()
            while (np.shape(y_train_selected)[0]+np.shape(df_new_fake_labels)[0]) <= max_class_val:
                for file in glob.glob("cgan_generator_b_" + str(bld) + "_f_" + str(flr) + "_i_*.h5"):
                    model = load_model(file, compile=False)
                    # Random generator
                    latent_points, labels = generate_latent_points(np.shape(X_train)[1],
                                                                   gan_general_config["num_fake_samples"])
                    # specify labels
                    labels = np.zeros(gan_general_config["num_fake_samples"])
                    # generate fingerprints
                    X = model.predict([latent_points, labels])
                    # Reshape
                    X_reshaped = np.reshape(X, (gan_general_config["num_fake_samples"], np.shape(X_train)[1]))
                    fake_fingerprints = pd.DataFrame(X_reshaped)

                    # Predict position, floor and building
                    X_train_series_nd = fake_fingerprints.values.reshape((fake_fingerprints.shape[0],
                                                                          fake_fingerprints.shape[1], 1))
                    subsequences = 2
                    timesteps = X_train_series_nd.shape[1] // subsequences
                    X_train_series_sub_nd = X_train_series_nd.reshape(
                        (X_train_series_nd.shape[0], subsequences, timesteps, 1))

                    longitude = longitude_model.predict(X_train_series_sub_nd)
                    latitude = latitude_model.predict(X_train_series_sub_nd)
                    floor = np.argmax(floor_model.predict(X_train_series_sub_nd), axis=-1)
                    building = np.argmax(building_model.predict(X_train_series_sub_nd), axis=-1)

                    predict_lat = scaler_latitude.inverse_transform(latitude[:, 0].reshape(-1, 1))
                    predict_long = scaler_longitude.inverse_transform(longitude[:, 0].reshape(-1, 1))
                    latitude = np.reshape(predict_lat[:], (1, len(predict_lat[:, 0])))
                    longitude = np.reshape(predict_long[:], (1, len(predict_long[:, 0])))

                    # Remove predicted fingerprints which are not in the building of analysis
                    index_bld_lab = [idx for idx, element in enumerate(building) if element == int(bld)]
                    floor = floor[[index_bld_lab]]
                    building = building[[index_bld_lab]]
                    latitude = latitude[0][[index_bld_lab]]
                    longitude = longitude[0][[index_bld_lab]]
                    fake_fingerprints = fake_fingerprints.loc[index_bld_lab, :].copy().reset_index(drop=True)

                    # Remove predicted fingerprints which are not in the floor of analysis
                    index_acc_lab = [idx for idx, element in enumerate(floor) if element == int(flr)]

                    floor = floor[[index_acc_lab]]
                    building = building[[index_acc_lab]]
                    latitude = latitude[[index_acc_lab]]
                    longitude = longitude[[index_acc_lab]]
                    fake_fingerprints = fake_fingerprints.loc[index_acc_lab, :].copy().reset_index(drop=True)

                    # Select realistic fingerprints
                    distance_matrix = np.zeros((np.shape(y_train_selected)[0], np.shape(longitude)[0]))

                    for i in range(0, (np.shape(distance_matrix)[0]) - 1):
                        for j in range(0, (np.shape(distance_matrix)[1]) - 1):
                            distance_matrix[i, j] = np.mean(np.sqrt(
                                np.square(longitude[j] - y_train_selected['LONGITUDE'].iloc[i]) +
                                np.square(latitude[j] - y_train_selected['LATITUDE'].iloc[i])))
                            if distance_matrix[i, j] < 5:
                                if (y_train['FLOOR'].iloc[i] != floor[j]) or (
                                        y_train['BUILDINGID'].iloc[i] != floor[j]):
                                    distance_matrix[i, j] = 1000000

                    distance_df = pd.DataFrame(distance_matrix)
                    filter = ((distance_df < 5) & (distance_df > 0)).any()
                    sub_df = distance_df.loc[:, filter]

                    new_data = list(zip(longitude[[sub_df.columns]], latitude[[sub_df.columns]],
                                        floor[[sub_df.columns]], building[[sub_df.columns]]))

                    df_new_fake_labels = df_new_fake_labels.append(new_data, ignore_index=True)
                    # Features X_train_new_data
                    df_new_fake_fingerprints = df_new_fake_fingerprints.append(fake_fingerprints.loc[filter, :],
                                                                               ignore_index=True)
                    print(np.shape(df_new_fake_fingerprints), flr)

            # Save new data
            df_full_augmented_y_train = df_full_augmented_y_train.append(df_new_fake_labels, ignore_index=True)
            df_full_augmented_x_train = df_full_augmented_x_train.append(df_new_fake_fingerprints, ignore_index=True)

    os.chdir('../../')
    df_full_augmented_y_train.columns = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']
    df_full_augmented_y_train.to_csv(source_path+'/TrainingData_y_augmented.csv', index_label=False, index=False)
    df_full_augmented_x_train.to_csv(source_path+'/TrainingData_x_augmented.csv', index_label=False, index=False)