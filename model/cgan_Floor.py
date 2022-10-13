import numpy as np
import os
from numpy import expand_dims
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from collections import Counter
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout

import matplotlib.pyplot as plt
from miscellaneous.misc import Misc
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### Warning ###
warnings.filterwarnings('ignore')

# For reproducibility
rnd_seed = 11
tf.random.set_seed(
    rnd_seed
)

gpu_available = tf.config.list_physical_devices('GPU')

if gpu_available:
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

'''
Based on: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
by: Jason Brownlee
'''

# GAN discriminator
def define_discriminator(X_train=None, y_train=None, discriminator_config=None):
    # label input
    in_shape = (np.shape(X_train)[1], 1)
    n_classes = int(max(y_train) + 1)
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((in_shape[0], 1))(li)
    in_dataset = Input(shape=in_shape)
    merge = Concatenate()([in_dataset, li])
    fe = Conv1D(filters=32, kernel_size=4)(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(64, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(128, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(256, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(512, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)

    output = Dense(1, activation='sigmoid')(fe)
    model = Model([in_dataset, in_label], output)
    opt = Adam(learning_rate=0.0002)
    misc = Misc()
    optimizer = misc.optimizer(discriminator_config['optimizer'], discriminator_config['lr'])
    model.compile(loss=discriminator_config['loss'], optimizer=optimizer, metrics=['accuracy'])
    return model

# GAN - Generator
def define_generator(X_train=None, y_train=None):
    # label input
    latent_dim = np.shape(X_train)[1]
    features = np.shape(X_train)[1]
    n_classes = int(max(y_train) + 1)

    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 1 * features
    li = Dense(n_nodes)(li)
    li = Reshape((features, 1))(li)
    in_lat = Input(shape=(latent_dim,))

    n_nodes = 1 * features

    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((features, 1))(gen)
    merge = Concatenate()([gen, li])
    gen = Conv1DTranspose(32, 4, padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv1DTranspose(64, 4, padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv1DTranspose(128, 4, padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv1DTranspose(1, 1, activation='tanh', padding='same')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model

# Building the gan model
def define_gan(g_model, d_model, gan_config=None):
    d_model.trainable = False
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    misc = Misc()
    optimizer = misc.optimizer(gan_config['optimizer'], gan_config['lr'])
    model.compile(loss=gan_config['loss'], optimizer=optimizer)
    return model

# Latent points generator
def generate_latent_points(latent_dim, n_samples, n_classes=1):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# Generating fake samples
def generate_fake_samples(generator, latent_dim, n_samples, n_classes=1):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
    datasets = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    return [datasets, labels_input], y


def data_selection(X_train=None, y_train=None, num_samples=10):
    ix = randint(0, X_train.shape[0], num_samples)
    X_train = X_train[ix, :]
    y_train = y_train[ix]
    y = ones((num_samples, 1))
    return X_train, y_train, y

# train the generator and discriminator
def train(X_train, y_train, iteration, g_model, d_model, gan_model, dataset_config=None, gan_general_config=None,
          path_config=None, floor=None, algorithm=None, method=None):
    print("Cleaning session...")
    K.clear_session()
    if gan_general_config['train'] == 'True':

        n_samples = gan_general_config['num_fake_samples']
        n_epochs = gan_general_config['epochs']
        latent_dim = np.shape(X_train)[1]
        n_batch = gan_general_config['batch_size']
        half_batch = int(n_batch / 2)
        bat_per_epo = int(np.shape(X_train)[0] / n_batch)
        misc = Misc()

        sub_path = "epochs_" + str(gan_general_config['epochs']) + '_bs_' + str(gan_general_config['batch_size'])
        main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'], algorithm, method,
                                      sub_path)

        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)
        iteration = str(iteration)

        if floor is None:
            floor = 0
        print(misc.log_msg('WARNING', "------- Floor " + str(floor) + " - Iteration " + iteration.zfill(2) + " -------"))
      
        for i in range(n_epochs):
            for j in range(bat_per_epo):
                X_real, labels_real, y_real = data_selection(X_train=X_train, y_train=y_train,
                                                             num_samples=half_batch)

                n_classes = max(y_train) + 1

                d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_classes)
                d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
                _, acc_real = d_model.evaluate([X_real, labels_real], y_real, verbose=0)
                _, acc_fake = d_model.evaluate([X_fake, labels], y_fake, verbose=0)
                [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, n_classes)
                y_gan = ones((n_batch, 1))
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

                print('>Epoch: %d, Batch epoch %d/%d, Discriminator real - Loss %.4f, '
                      'Discriminator fake loss %.4f, GAN loss %.4f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

        g_model.save(main_path_save + '/cgan_generator_floor_' + str(gan_general_config['epochs']) + '_' +
                     str(gan_general_config['batch_size']) + '_floor_' + str(floor) + '_iteration_' + iteration.zfill(2)
                     + '.h5')


def train_imbalance_classes_floor(X_train, y_train, dataset_config=None, discriminator_config=None,
                                  gan_general_config=None, gan_config=None, path_config=None, algorithm=None,
                                  method=None):
    list_candidates = np.unique(y_train['FLOOR'])
    
    for idx_class in list_candidates:
        index_samples = y_train[y_train['FLOOR'] == idx_class].index
        X_train_selected = X_train[index_samples, :]
        y_train_selected = y_train[y_train['FLOOR'] == idx_class].values
        X_train_segment = X_train_selected
        X_train_segment = X_train_segment.reshape((X_train_segment.shape[0], X_train_segment.shape[1], 1))
        y_train_segment = y_train_selected[:, 4]  # 3-Floor, 4-Building

        # create the discriminator    
        d_model = define_discriminator(X_train=X_train_segment, y_train=y_train_segment,
                                       discriminator_config=discriminator_config)
        # create the generator
        g_model = define_generator(X_train=X_train_segment, y_train=y_train_segment)
        # create the gan
        gan_model = define_gan(g_model, d_model, gan_config=gan_config)
        # train model
        train(X_train_segment, y_train_segment, 0, g_model, d_model, gan_model, dataset_config=dataset_config,
              gan_general_config=gan_general_config, path_config=path_config, floor=idx_class,
              algorithm=algorithm, method=method)