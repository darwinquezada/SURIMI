import pandas as pd
from numpy import array
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from numpy.random import seed
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
seed(rnd_seed)
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
                 path_config, building_config, floor_config, position_config, algorithm):
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

    def building_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.bl_model = Sequential()
        self.bl_model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'),
                                          input_shape=(None, X_train.shape[2], X_train.shape[3]))) # 64, 1
        self.bl_model.add(TimeDistributed(MaxPooling1D(pool_size=2))) # 1
        #self.bl_model.add(TimeDistributed(Conv1D(16, kernel_size=8, activation='relu')))  # New 32, 8
        self.bl_model.add(TimeDistributed(Dropout(0.8))) # New
        self.bl_model.add(TimeDistributed(Flatten()))
        self.bl_model.add(LSTM(40, activation='relu')) # 40
        self.bl_model.add(Dense(self.classes_bld, activation='softmax'))

    def floor_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.fl_model = Sequential()
        self.fl_model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'),
                                          input_shape=(None, X_train.shape[2], X_train.shape[3])))  #16,1
        self.fl_model.add(TimeDistributed(MaxPooling1D(pool_size=1)))  # 1
        self.fl_model.add(TimeDistributed(Conv1D(32, kernel_size=1, activation='relu')))
        # self.fl_model.add(TimeDistributed(Conv1D(32, kernel_size=1, activation='relu',
        #                                          kernel_regularizer=regularizers.l1(l1=1e-2)))) # 32, l1=1e-2
        # self.fl_model.add(TimeDistributed(MaxPooling1D(pool_size=1))) # New
        self.fl_model.add(TimeDistributed(Dropout(0.5)))
        self.fl_model.add(TimeDistributed(Flatten()))
        self.fl_model.add(LSTM(50, activation='relu'))  # 50
        self.fl_model.add(Dense(self.classes_floor, activation='softmax'))

    def position_lati_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.model_lat = Sequential()
        self.model_lat.add(TimeDistributed(Conv1D(filters=90, kernel_size=12, activation='elu'),
                                           input_shape=(None, X_train.shape[2], X_train.shape[3])))  # 90, 12
        self.model_lat.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model_lat.add(TimeDistributed(Dropout(0.5)))
        self.model_lat.add(TimeDistributed(Flatten()))
        self.model_lat.add(LSTM(50, activation='elu'))  # 50
        self.model_lat.add(Dense(1))

    def position_long_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.model_long = Sequential()
        self.model_long.add(TimeDistributed(Conv1D(filters=90, kernel_size=12, activation='elu'),
                                            input_shape=(None, X_train.shape[2], X_train.shape[3])))  # 90, 12
        self.model_long.add(TimeDistributed(MaxPooling1D(pool_size=2)))  # 2
        self.model_long.add(TimeDistributed(Dropout(0.5)))
        self.model_long.add(TimeDistributed(Flatten()))
        self.model_long.add(LSTM(40, activation='elu'))  # 50
        self.model_long.add(Dense(1))

    def position_alt_model(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)
        self.model_alt = Sequential()
        self.model_alt.add(TimeDistributed(Conv1D(filters=68, kernel_size=12, activation='elu'),
                                           input_shape=(None, X_train.shape[2], X_train.shape[3]))) # 128, 64
        self.model_alt.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model_alt.add(TimeDistributed(Dropout(0.5)))
        self.model_alt.add(TimeDistributed(Flatten()))
        self.model_alt.add(LSTM(50, activation='elu'))   # 50
        self.model_alt.add(Dense(1, activation='relu'))

    def train(self):
        X_train, X_test, X_valid = data_reshape_stf(self.X_train, self.X_test, self.X_valid)

        # EarlyStopping
        early_stopping = EarlyStopping(monitor='loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto') # val_loss
        misc = Misc()
        main_path_save = os.path.join(self.path_config['saved_model'], self.dataset_config['name'], self.algorithm)
        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)
        # ---------------------------------------- Building ------------------------------------------
        if (self.building_config['train']).upper() == 'TRUE':
            print(misc.log_msg("WARNING", "--------- BUILDING CLASSIFICATION -----------"))
            bld_encoder = OneHotEncoder(sparse=False)
            y_train_bld = bld_encoder.fit_transform(self.y_train[:, 4].reshape(-1, 1))

            # Save encoder model
            joblib.dump(bld_encoder, main_path_save + '/building_onehotencoder.save')

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

            # Plot loss vs. val_loss
            # plt.plot(bld_history.history['loss'])
            # plt.plot(bld_history.history['val_loss'])
            # plt.title(self.dataset_config['name'] + ' Model loss - Building')
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.legend(['Train', 'Validation'], loc='upper left')
            # plt.show()
        else:
            self.bl_model = load_model(main_path_save + '/building.h5')

        # ---------------------------------------- Floor --------------------------------------------
        if(self.floor_config['train']).upper() == 'TRUE':
            print(misc.log_msg("WARNING", "--------- FLOOR CLASSIFICATION -----------"))
            # Encoding labels
            floor_encoder = OneHotEncoder(sparse=False)
            y_train_fl = floor_encoder.fit_transform(self.y_train[:, 3].reshape(-1, 1))

            # Save encoder model
            joblib.dump(floor_encoder, main_path_save + '/floor_onehotencoder.save')

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

            # Plot loss vs. val_loss
            # plt.plot(floor_history.history['loss'])
            # plt.plot(floor_history.history['val_loss'])
            # plt.title(self.dataset_config['name'] + ' Model loss - Floor')
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.legend(['Train', 'Validation'], loc='upper left')
            # plt.show()
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

            self.position_long_model()
            optimizer_long = misc.optimizer(self.position_config['optimizer_longitude'],
                                            self.position_config['lr_longitude'])
            self.model_long.compile(loss=self.position_config['loss_longitude'], optimizer=optimizer_long)

            self.position_lati_model()
            optimizer_lati = misc.optimizer(self.position_config['optimizer_latitude'],
                                            self.position_config['lr_latitude'])
            self.model_lat.compile(loss=self.position_config['loss_latitude'], optimizer=optimizer_lati)

            self.position_alt_model()
            optimizer_alti = misc.optimizer(self.position_config['optimizer_altitude'],
                                            self.position_config['lr_altitude'])
            self.model_alt.compile(loss=self.position_config['loss_altitude'], optimizer=optimizer_alti)

            if np.size(X_valid) == 0:
                long_history = self.model_long.fit(X_train, long_y_train,
                                                   epochs=self.position_config['epochs_longitude'], verbose=1,
                                                   callbacks=[early_stopping])
                lati_history = self.model_lat.fit(X_train, lat_y_train,
                                                  epochs=self.position_config['epochs_latitude'], verbose=1,
                                                  callbacks=[early_stopping])
                alti_history = self.model_alt.fit(X_train, alt_y_train,
                                                  epochs=self.position_config['epochs_altitude'], verbose=1,
                                                  callbacks=[early_stopping])
            else:
                long_y_valid = scaler_long.transform(self.y_valid[:, 0].reshape(-1, 1))
                lat_y_valid = scaler_lat.transform(self.y_valid[:, 1].reshape(-1, 1))
                alt_y_valid = scaler_alt.transform(self.y_valid[:, 2].reshape(-1, 1))
                long_history = self.model_long.fit(X_train, long_y_train,
                                                   validation_data=(X_valid, long_y_valid),
                                                   epochs=self.position_config['epochs_longitude'], verbose=1,
                                                   callbacks=[early_stopping])
                lati_history = self.model_lat.fit(X_train, lat_y_train,
                                                  validation_data=(X_valid, lat_y_valid),
                                                  epochs=self.position_config['epochs_latitude'], verbose=1,
                                                  callbacks=[early_stopping])
                alti_history = self.model_alt.fit(X_train, alt_y_train,
                                                  validation_data=(X_valid, alt_y_valid),
                                                  epochs=self.position_config['epochs_altitude'], verbose=1,
                                                  callbacks=[early_stopping])

            self.model_long.save(main_path_save + '/pos-long.h5')
            self.model_lat.save(main_path_save + '/pos-lati.h5')
            self.model_alt.save(main_path_save + '/pos-alti.h5')

            # Plot loss vs. val_loss
            '''
            plt.plot(long_history.history['loss'])
            plt.plot(long_history.history['val_loss'])
            plt.title(self.dataset_config['name'] + ' Model loss - Longitude')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()

            plt.plot(lati_history.history['loss'])
            plt.plot(lati_history.history['val_loss'])
            plt.title(self.dataset_config['name'] + ' Model loss - Latitude')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()

            plt.plot(alti_history.history['loss'])
            plt.plot(alti_history.history['val_loss'])
            plt.title(self.dataset_config['name'] + ' Model loss - Altitude')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
            '''
        else:
            scaler_lat = joblib.load(main_path_save+'/lati_minmaxscaler.save')
            scaler_long = joblib.load(main_path_save+'/long_minmaxscaler.save')
            scaler_alt = joblib.load(main_path_save+'/alti_minmaxscaler.save')
            self.model_long = load_model(main_path_save + '/pos-long.h5')
            self.model_lat = load_model(main_path_save + '/pos-lati.h5')
            self.model_alt = load_model(main_path_save + '/pos-alti.h5')

        # --------------------------------------- Predicting ---------------------------------
        # Predict Building
        predicted_bld = self.bl_model.predict(X_test)
        predicted_bld = np.argmax(predicted_bld, axis=-1)

        # Predict floor
        predicted_floor = self.fl_model.predict(X_test)
        predicted_floor = np.argmax(predicted_floor, axis=-1)

        # Predict position
        predicted_latitude = self.model_lat.predict(X_test)
        predicted_longitude = self.model_long.predict(X_test)
        predicted_altitude = self.model_alt.predict(X_test)

        predict_lat = scaler_lat.inverse_transform(predicted_latitude[:, 0].reshape(-1, 1))
        predict_long = scaler_long.inverse_transform(predicted_longitude[:, 0].reshape(-1, 1))
        predict_alt = scaler_alt.inverse_transform(predicted_altitude[:, 0].reshape(-1, 1))

        predict_lat = np.reshape(predict_lat[:], (1, len(predict_lat[:, 0])))
        predict_long = np.reshape(predict_long[:], (1, len(predict_long[:, 0])))
        predict_alt = np.reshape(predict_alt[:], (1, len(predict_alt[:, 0])))

        df_prediction = pd.DataFrame(list(zip(predict_long[0][:], predict_lat[0][:], predict_alt[0][:], predicted_floor,
                                              predicted_bld)),
                                     columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDING'])
        prediction_path = os.path.join(self.path_config['results'], self.dataset_config['name'], self.algorithm)
        # Save prediction
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        df_prediction.to_csv(prediction_path+'/prediction.csv', header=True, index=False)
        return df_prediction

    def test_model(self, dataset, x_test):
        '''
        Prueba del modelo

        Parámetros de ingreso:
        :param string dataset: nombre del dataset
        :param Matrix X_test: datos de test
        :return preds: Predicción del modelo
        '''
        dir_h5 = os.path.join("./saved_models", dataset, "h5")
        model_loaded = load_model(dir_h5 + '/cnn-' + dataset + '.h5')
        model_loaded.summary()
        preds = model_loaded.predict(x_test)
        return preds

    def eval_model(self, x_test):
        preds = self.cnn_model.predict(x_test)
        return preds
