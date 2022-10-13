import pandas as pd
from numpy import array
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from numpy.random import seed, default_rng
import numpy as np
from miscellaneous.misc import Misc
import joblib

from preprocessing.data_preprocessing import data_reshape_stf, data_reshape_st

import matplotlib.pyplot as plt

### Warning ###
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
rnd_seed = 11
default_rng(rnd_seed)
tf.random.set_seed(
    rnd_seed
)

gpu_available = tf.test.is_gpu_available()

if gpu_available:
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


class CNN_LSTM():
    def __init__(self, X_train, y_train, X_test, y_test, X_valid, y_valid, dataset_config,
                 path_config, building_config, floor_config, position_config, algorithm, **kwargs):
        """

        Parameters
        ----------
        X_train : Training set
        y_train : Labels training set
        X_test : Test set
        y_test : Labels test set
        X_valid : Validation set
        y_valid : Labels validation set
        dataset_config : Dataset config
        path_config : General paths set in the config file
        building_config : Hyperparameters building model (config file)
        floor_config : Hyperparameters floor model (config file)
        position_config : Hyperparameters positioning model (config file)
        algorithm : Algorithm used
        kwargs : parameters of gan_general_config, method, data_augmentation
        """
        loss = 'mse'
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.dataset_config = dataset_config
        self.building_config = building_config
        self.floor_config = floor_config
        self.position_config = position_config
        self.path_config = path_config
        self.algorithm = algorithm
        self.classes_floor = np.shape(np.unique(self.y_train[:, 3]))[0]
        self.classes_bld = np.shape(np.unique(self.y_train[:, 4]))[0]
        self.gan_general_conf = None
        self.data_augmentation = None
        self.method = None

        for key, value in kwargs.items():
            if key == 'gan_general_config':
                self.gan_general_conf = value

            if key == 'method':
                self.method = value

            if key == 'data_augmentation':
                self.data_augmentation = value
                print(self.data_augmentation)

    # Model to classify the fingerprints into buildings
    def building_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.bl_model = Sequential()
        self.bl_model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'),
                                          input_shape=(None, X_train.shape[2], X_train.shape[3]))) 
        self.bl_model.add(TimeDistributed(MaxPooling1D(pool_size=2))) # 
        self.bl_model.add(TimeDistributed(Dropout(0.8))) 
        self.bl_model.add(TimeDistributed(Flatten()))
        self.bl_model.add(LSTM(40, activation='relu')) 
        self.bl_model.add(Dense(self.classes_bld, activation='softmax'))

    # Model to classify the fingerprints into floor
    def floor_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.fl_model = Sequential()
        self.fl_model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'),
                                          input_shape=(None, X_train.shape[2], X_train.shape[3]))) 
        self.fl_model.add(TimeDistributed(MaxPooling1D(pool_size=2)))  # 1
        self.fl_model.add(TimeDistributed(Dropout(0.5)))
        self.fl_model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu',
                                                 padding='same')))
        self.fl_model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.fl_model.add(TimeDistributed(Dropout(0.5)))
        self.fl_model.add(TimeDistributed(Flatten()))
        self.fl_model.add(LSTM(50, activation='relu'))
        self.fl_model.add(Dense(self.classes_floor, activation='softmax'))

    # Regression model to predict the position (x,y,z)
    def position_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.model_pos = Sequential()
        self.model_pos.add(TimeDistributed(Conv1D(filters=8, kernel_size=1, activation='elu'),
                                           input_shape=(None, X_train.shape[2], X_train.shape[3])))
        self.model_pos.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model_pos.add(TimeDistributed(Dropout(0.5)))
        self.model_pos.add(TimeDistributed(Conv1D(filters=8, kernel_size=1, activation='elu', padding='same')))
        self.model_pos.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model_pos.add(TimeDistributed(Dropout(0.5)))
        self.model_pos.add(TimeDistributed(Flatten()))
        self.model_pos.add(LSTM(40, activation='elu'))
        self.model_pos.add(Dense(3, activation='elu'))

    def train(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)

        if np.size(X_valid) == 0:
            monitor = 'loss'
        else:
            monitor = 'val_loss'

        # EarlyStopping
        early_stopping = EarlyStopping(monitor=monitor,
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')  # val_loss
        misc = Misc()

        if self.gan_general_conf is not None:
            string_dist = [str(dist) for dist in self.data_augmentation['distance_rsamples']]
            conf_augment = 'epochs_' + str(self.gan_general_conf['epochs']) + '_bs_' + str(
                self.gan_general_conf['batch_size']) + \
                           '_dist_(' + ','.join(string_dist) + ')_iter_' + str(self.data_augmentation['iterations'])

            main_path_save = os.path.join(self.path_config['saved_model'], self.dataset_config['name'], self.algorithm,
                                          self.method, conf_augment)
            prediction_path = os.path.join(self.path_config['results'], self.dataset_config['name'], self.algorithm,
                                           self.method, conf_augment)
        else:
            main_path_save = os.path.join(self.path_config['saved_model'], self.dataset_config['name'], self.algorithm)
            prediction_path = os.path.join(self.path_config['results'], self.dataset_config['name'], self.algorithm)
        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)

        # ---------------------------------------- Building ------------------------------------------
        if (self.building_config['train']).upper() == 'TRUE':
            print(misc.log_msg("WARNING", "--------- BUILDING CLASSIFICATION -----------"))

            # Label encoder
            bld_label_encoder = LabelEncoder()
            y_train_lab_enc = bld_label_encoder.fit_transform(self.y_train[:, 4])

            bld_encoder = OneHotEncoder(sparse=False)
            y_train_bld = bld_encoder.fit_transform(y_train_lab_enc.reshape(-1, 1))

            # Save encoder model
            joblib.dump(bld_encoder, main_path_save + '/building_onehotencoder.save')
            joblib.dump(bld_label_encoder, main_path_save + '/building_labelencoder.save')

            self.building_model()
            optimizer = misc.optimizer(self.building_config['optimizer'], self.building_config['lr'])
            self.bl_model.compile(loss=self.building_config['loss'], optimizer=optimizer) #, metrics=['accuracy'])

            if np.size(X_valid) == 0:
                bld_history = self.bl_model.fit(X_train, y_train_bld, epochs=self.building_config['epochs'], verbose=1,
                                                callbacks=[early_stopping])
            else:
                y_valid_bld = bld_encoder.transform(self.y_valid[:, 4].reshape(-1, 1))
                bld_history = self.bl_model.fit(X_train, y_train_bld, validation_data=(X_valid, y_valid_bld),
                                                epochs=self.building_config['epochs'], verbose=1,
                                                callbacks=[early_stopping])

            # Save model
            self.bl_model.save(main_path_save + '/building.h5')

        else:
            self.bl_model = load_model(main_path_save + '/building.h5')

        # ---------------------------------------- Floor --------------------------------------------
        if(self.floor_config['train']).upper() == 'TRUE':
            print(misc.log_msg("WARNING", "--------- FLOOR CLASSIFICATION -----------"))
            # Encoding label
            floor_label_encoder = LabelEncoder()
            y_train_fl = floor_label_encoder.fit_transform(self.y_train[:, 3])
            # Encoding labels
            floor_encoder = OneHotEncoder(sparse=False)
            y_train_fl = floor_encoder.fit_transform(y_train_fl.reshape(-1, 1))

            # Save encoder model
            joblib.dump(floor_encoder, main_path_save + '/floor_onehotencoder.save')
            joblib.dump(floor_label_encoder, main_path_save + '/floor_labelencoder.save')

            self.floor_model()
            optimizer = misc.optimizer(self.floor_config['optimizer'], self.floor_config['lr'])
            self.fl_model.compile(loss=self.floor_config['loss'], optimizer=optimizer) #, metrics=['accuracy'])

            if np.size(X_valid) == 0:
                floor_history = self.fl_model.fit(X_train, y_train_fl, epochs=self.floor_config['epochs'], verbose=1,
                                                  callbacks=[early_stopping])
            else:
                y_valid_fl = floor_encoder.transform(self.y_valid[:, 3].reshape(-1, 1))
                floor_history = self.fl_model.fit(X_train, y_train_fl, validation_data=(X_valid, y_valid_fl),
                                                  epochs=self.floor_config['epochs'], verbose=1,
                                                  callbacks=[early_stopping])

            # Save model
            self.fl_model.save(main_path_save+'/floor.h5')
        else:
            self.fl_model = load_model(main_path_save + '/floor.h5')

        # --------------------------- Position (Latitude, Longitude and altitude) ----------------------
        if (self.position_config['train']).upper() == 'TRUE':
            print(misc.log_msg("WARNING", "------- LONGITUDE, LATITUDE and ALTITUDE PREDICTION -------"))
            # Scale longitude
            scaler_long = MinMaxScaler()
            long_y_train = scaler_long.fit_transform(self.y_train[:, 0].reshape(-1, 1))
            # Save MinMaxScaler
            joblib.dump(scaler_long, main_path_save+'/long_minmaxscaler.save')

            # Scale Latitude
            scaler_lat = MinMaxScaler()
            lat_y_train = scaler_lat.fit_transform(self.y_train[:, 1].reshape(-1, 1))
            # Save MinMaxScaler
            joblib.dump(scaler_lat, main_path_save + '/lati_minmaxscaler.save')

            # Scale Altitude
            scaler_alt = MinMaxScaler()
            alt_y_train = scaler_alt.fit_transform(self.y_train[:, 2].reshape(-1, 1))
            # Save MinMaxScaler
            joblib.dump(scaler_alt, main_path_save + '/alti_minmaxscaler.save')

            self.position_model()
            optimizer = misc.optimizer(self.position_config['optimizer'],
                                       self.position_config['lr'])
            self.model_pos.compile(loss=self.position_config['loss'], optimizer=optimizer)

            train_data = np.hstack([long_y_train, lat_y_train, alt_y_train])
            if np.size(X_valid) == 0:
                pos_history = self.model_pos.fit(X_train, train_data, epochs=self.position_config['epochs'],
                                                 verbose=1, callbacks=[early_stopping])

            else:
                long_y_valid = scaler_long.transform(self.y_valid[:, 0].reshape(-1, 1))
                lat_y_valid = scaler_lat.transform(self.y_valid[:, 1].reshape(-1, 1))
                alt_y_valid = scaler_alt.transform(self.y_valid[:, 2].reshape(-1, 1))
                valid_data = np.hstack([long_y_valid, lat_y_valid, alt_y_valid])
                pos_history = self.model_pos.fit(X_train, train_data,
                                                 validation_data=(X_valid, valid_data),
                                                 epochs=self.position_config['epochs'],
                                                 verbose=1, callbacks=[early_stopping])

            self.model_pos.save(main_path_save + '/position.h5')

        else:
            bld_label_encoder = joblib.load(main_path_save+'/building_labelencoder.save')
            floor_label_encoder = joblib.load(main_path_save+'/floor_labelencoder.save')
            scaler_lat = joblib.load(main_path_save+'/lati_minmaxscaler.save')
            scaler_long = joblib.load(main_path_save+'/long_minmaxscaler.save')
            scaler_alt = joblib.load(main_path_save+'/alti_minmaxscaler.save')
            self.model_pos = load_model(main_path_save + '/position.h5')
        # --------------------------------------- Predicting ---------------------------------
        # Predict Building
        predicted_bld = self.bl_model.predict(X_test)
        predicted_bld = np.argmax(predicted_bld, axis=-1)
        predicted_bld = bld_label_encoder.inverse_transform(predicted_bld)

        # Predict floor
        predicted_floor = self.fl_model.predict(X_test)
        predicted_floor = np.argmax(predicted_floor, axis=-1)
        predicted_floor = floor_label_encoder.inverse_transform(predicted_floor)

        # Predict position
        predict_position = self.model_pos.predict(X_test)

        predict_long = scaler_long.inverse_transform(predict_position[:, 0].reshape(-1, 1))
        predict_lat = scaler_lat.inverse_transform(predict_position[:, 1].reshape(-1, 1))
        predict_alt = scaler_alt.inverse_transform(predict_position[:, 2].reshape(-1, 1))

        predict_long = np.reshape(predict_long[:], (1, len(predict_long[:, 0])))
        predict_lat = np.reshape(predict_lat[:], (1, len(predict_lat[:, 0])))
        predict_alt = np.reshape(predict_alt[:], (1, len(predict_alt[:, 0])))

        df_prediction = pd.DataFrame(list(zip(predict_long[0][:], predict_lat[0][:], predict_alt[0][:], predicted_floor,
                                              predicted_bld)),
                                     columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDING'])

        # Save prediction
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        df_prediction.to_csv(prediction_path+'/prediction.csv', header=True, index=False)
        return df_prediction

    def test_model(self, dataset, x_test):
        dir_h5 = os.path.join("./saved_models", dataset, "h5")
        model_loaded = load_model(dir_h5 + '/cnn-' + dataset + '.h5')
        model_loaded.summary()
        preds = model_loaded.predict(x_test)
        return preds

    def eval_model(self, x_test):
        preds = self.cnn_model.predict(x_test)
        return preds
