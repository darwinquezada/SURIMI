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


def generation(dataset_name=None, dataset_config=None, path_config=None, gan_general_config=None,
               data_augmentation=None, algorithm=None, method=None):
    source_path = os.path.join(path_config['data_source'], dataset_name)
    saved_model_path = os.path.join(path_config['saved_model'], dataset_name, 'CNN-LSTM')

    # Load saved models
    position_model = load_model(saved_model_path + '/position.h5', compile=False)
    floor_model = load_model(saved_model_path + '/floor.h5', compile=False)
    building_model = load_model(saved_model_path + '/building.h5', compile=False)

    scaler_latitude = joblib.load(saved_model_path + '/lati_minmaxscaler.save')
    scaler_longitude = joblib.load(saved_model_path + '/long_minmaxscaler.save')
    scaler_altitude = joblib.load(saved_model_path + '/alti_minmaxscaler.save')
    floor_label_scaler = joblib.load(saved_model_path + '/floor_labelencoder.save')
    building_label_scaler = joblib.load(saved_model_path + '/building_labelencoder.save')

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

    # Data Normalization
    new_non_det_val = new_non_detected_value(X_train, X_test, X_valid)
    dr = DataRepresentation(x_train=X_train, x_test=X_test, x_valid=X_valid,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()

    files = []
    sub_path = "epochs_" + str(gan_general_config['epochs']) + '_bs_' + str(gan_general_config['batch_size'])
    saved_gan_model = os.path.join(path_config['saved_model'], dataset_name, algorithm, method, sub_path)

    if not os.path.exists(saved_gan_model):
        os.makedirs(saved_gan_model)

    os.chdir(saved_gan_model)

    for file in glob.glob("cgan_generator_*.h5"):
        files.append(file)

    df_full_augmented_y_train = pd.DataFrame()
    df_full_augmented_x_train = pd.DataFrame()

    for file in files:
        print(misc.log_msg("WARNING", "-----------------------------"))
        print(misc.log_msg("ERROR", "Loading model: " + file))
        model = load_model(file, compile=False)

        df_new_fake_labels = pd.DataFrame()
        df_new_fake_fingerprints = pd.DataFrame()

        for dis in data_augmentation['distance_rsamples']:
            for i in range(0, data_augmentation['iterations']):
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
                if np.shape(fake_fingerprints)[1] % 2 != 0:
                    fake_fingerprints = np.append(fake_fingerprints.values,
                                                  np.zeros((np.shape(fake_fingerprints)[0], 1)), axis=1)
                    fake_fingerprints = pd.DataFrame(fake_fingerprints)

                X_train_series_nd = fake_fingerprints.values.reshape((np.shape(fake_fingerprints)[0],
                                                                      np.shape(fake_fingerprints)[1], 1))
                subsequences = 2
                timesteps = X_train_series_nd.shape[1] // subsequences
                X_train_series_sub_nd = X_train_series_nd.reshape(
                    (X_train_series_nd.shape[0], subsequences, timesteps, 1))

                position = position_model.predict(X_train_series_sub_nd)
                floor = np.argmax(floor_model.predict(X_train_series_sub_nd), axis=-1)
                building = np.argmax(building_model.predict(X_train_series_sub_nd), axis=-1)

                floor = floor_label_scaler.inverse_transform(floor)
                building = building_label_scaler.inverse_transform(building)

                predict_long = scaler_longitude.inverse_transform(position[:, 0].reshape(-1, 1))
                predict_lat = scaler_latitude.inverse_transform(position[:, 1].reshape(-1, 1))
                predict_alt = scaler_altitude.inverse_transform(position[:, 2].reshape(-1, 1))

                latitude = np.reshape(predict_lat[:], (1, len(predict_lat[:, 0])))
                longitude = np.reshape(predict_long[:], (1, len(predict_long[:, 0])))
                altitude = np.reshape(predict_alt[:], (1, len(predict_alt[:, 0])))

                # Select realistic fingerprints

                uni_pred_buildings = np.unique(building)

                for bld in uni_pred_buildings:
                    bld_segment_y_train = y_train[y_train['BUILDINGID'] == int(bld)].reset_index(drop=True).copy()

                    index_pred = np.where(building == int(bld))
                    uni_pred_floor = np.unique(floor[index_pred])

                    bld_sel_lat = latitude[0][index_pred]
                    bld_sel_lon = longitude[0][index_pred]
                    bld_sel_alt = altitude[0][index_pred]
                    bld_sel_flr = floor[index_pred]
                    bld_sel_bld = building[index_pred]

                    bld_fake_fp = fake_fingerprints.loc[index_pred[0], :].copy().reset_index(drop=True)

                    for flr in uni_pred_floor:
                        sel_y_train = bld_segment_y_train[bld_segment_y_train['FLOOR'] == int(flr)].\
                            reset_index(drop=True).copy()

                        idx_flr = np.where(bld_sel_flr == int(flr))
                        flr_sel_lat = bld_sel_lat[idx_flr]
                        flr_sel_lon = bld_sel_lon[idx_flr]
                        flr_sel_alt = bld_sel_alt[idx_flr]
                        flr_sel_flr = bld_sel_flr[idx_flr]
                        flr_sel_bld = bld_sel_bld[idx_flr]
                        flr_fake_fp = bld_fake_fp.loc[idx_flr[0], :].copy().reset_index(drop=True)

                        # Select realistic fingerprints
                        distance_matrix = np.zeros((np.shape(sel_y_train)[0], len(flr_sel_bld)))

                        for tr in range(0, (np.shape(distance_matrix)[0]) - 1):
                            for pr in range(0, (np.shape(distance_matrix)[1]) - 1):
                                distance_matrix[tr, pr] = np.mean(np.sqrt(
                                    np.square(flr_sel_lon[pr] - sel_y_train['LONGITUDE'].iloc[tr]) +
                                    np.square(flr_sel_lat[pr] - sel_y_train['LATITUDE'].iloc[tr]) +
                                    np.square(flr_sel_alt[pr] - sel_y_train['ALTITUDE'].iloc[tr])
                                ))
                                alti_error = np.sqrt(np.square(flr_sel_alt[pr] - sel_y_train['ALTITUDE'].iloc[tr]))
                                if alti_error > 0.2:
                                    distance_matrix[tr, pr] = 1000000

                        distance_df = pd.DataFrame(distance_matrix)
                        filter = ((distance_df < dis) & (distance_df > 0)).any()
                        sub_df = distance_df.loc[:, filter]

                        new_data = list(zip(flr_sel_lon[[sub_df.columns]], flr_sel_lat[[sub_df.columns]],
                                            flr_sel_alt[[sub_df.columns]], flr_sel_flr[[sub_df.columns]],
                                            flr_sel_bld[[sub_df.columns]]))

                        df_new_fake_labels = df_new_fake_labels.append(new_data, ignore_index=True)
                        # Features X_train_new_data

                        df_new_fake_fingerprints = df_new_fake_fingerprints.append(flr_fake_fp.loc[filter, :],
                                                                                   ignore_index=True)


            # Save new data
            df_full_augmented_y_train = df_full_augmented_y_train.append(df_new_fake_labels, ignore_index=True)
            df_full_augmented_x_train = df_full_augmented_x_train.append(df_new_fake_fingerprints, ignore_index=True)

    os.chdir('../../../../../')

    if np.shape(df_full_augmented_y_train)[0] > 0:
        print(">Samples: %d" %
              (np.shape(df_full_augmented_y_train)[0]))

        saved_new_fp = os.path.join(path_config['saved_model'], dataset_name, algorithm)
        string_dist = [str(dist) for dist in data_augmentation['distance_rsamples']]
        conf_augment = 'epochs_' + str(gan_general_config['epochs']) + '_bs_' + str(gan_general_config['batch_size']) + \
                       '_dist_(' + ','.join(string_dist) + ')_iter_' + str(data_augmentation['iterations'])
        destination_folder = os.path.join(saved_new_fp, method, conf_augment)

        df_full_augmented_y_train.columns = ['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID']
        df_full_augmented_y_train.to_csv(destination_folder + '/TrainingData_y_augmented.csv', index_label=False,
                                         index=False)
        df_full_augmented_x_train.to_csv(destination_folder + '/TrainingData_x_augmented.csv', index_label=False,
                                         index=False)
        print("Shape New Fingerprints")
        print(np.shape(df_full_augmented_y_train))
    else:
        print(misc.log_msg("ERROR", "Oops, something went wrong, no fake fingerprints."))
        exit(-1)

    return df_full_augmented_x_train, df_full_augmented_y_train
