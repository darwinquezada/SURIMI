# example of training an conditional gan on the fashion mnist dataset
import numpy as np
import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from collections import Counter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import RandomNormal
from miscellaneous.misc import Misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# define the standalone discriminator model
def define_discriminator(X_train=None, y_train=None, discriminator_config=None):
    # label input
    in_shape = (np.shape(X_train)[1], 1)
    n_classes = int(max(y_train) + 1)
    in_label = Input(shape=(1,))

    # weight initialization
    init = RandomNormal(stddev=0.02)

    in_dataset = Input(shape=in_shape)
    fe = Conv1D(filters=32, kernel_size=4, padding='same')(in_dataset)
    fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.5)(fe)
    fe = Conv1D(64, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.5)(fe)
    fe = Conv1D(128, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.5)(fe)
    fe = Conv1D(256, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(512, kernel_size=4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.5)(fe)
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)

    output = Dense(1, activation='sigmoid')(fe)
    output2 = Dense(n_classes, activation='softmax')(fe)
    model = Model(in_dataset, [output, output2])

    misc = Misc()
    optimizer = misc.optimizer(discriminator_config['optimizer'], discriminator_config['lr'])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# define the standalone generator model
def define_generator(X_train=None, y_train=None):
    # label input
    latent_dim = np.shape(X_train)[1]
    features = np.shape(X_train)[1]
    n_classes = int(max(y_train) + 1)

    init = RandomNormal(stddev=0.02)

    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 1 * features
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((features, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))

    n_nodes = 1 * features

    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((features, 1))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    gen = Conv1DTranspose(32, 4, padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # gen = Dropout(0.5)(gen)
    gen = Conv1DTranspose(64, 4, padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # gen = Dropout(0.5)(gen)
    gen = Conv1DTranspose(128, 4, padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # gen = Dropout(0.5)(gen)
    # output
    out_layer = Conv1DTranspose(1, 1, activation='tanh', padding='same')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, gan_config=None):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=1):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes=1):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


def data_selection(X_train=None, y_train=None, num_samples=10):
    ix = randint(0, X_train.shape[0], num_samples)
    # select fingerprints and labels
    X_train = X_train[ix, :]
    y_train = y_train[ix]
    # generate class labels
    y = ones((num_samples, 1))
    return X_train, y_train, y


# train the generator and discriminator
def train(X_train, y_train, iteration, g_model, d_model, gan_model, dataset_config=None, gan_general_config=None,
          path_config=None):
    if gan_general_config['train'] == 'True':
        n_samples = gan_general_config['num_fake_samples']
        n_epochs = gan_general_config['epochs']
        latent_dim = np.shape(X_train)[1]
        n_batch = gan_general_config['batch_size']
        bat_per_epo = int(np.shape(X_train)[0] / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)

        misc = Misc()
        main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'])

        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)

        for i in range(n_steps):
            # get randomly selected 'real' samples
            X_real, labels_real, y_real = data_selection(X_train=X_train, y_train=y_train,
                                                         num_samples=half_batch)
            # update discriminator model weights
            _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
            # generate 'fake' examples
            [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
            # prepare points in latent space as input for the generator
            [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
            # summarize loss on this batch
            print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
            if (i + 1) % (bat_per_epo * 10) == 0:
                print(misc.log_msg("WARNING", "Saving model...."))
                g_model.save(main_path_save + '/acgan_generator_full_db_' + str(gan_general_config['epochs']) + '_'
                             + str(gan_general_config['batch_size']) + '.h5')


def train_imbalance_classes(X_train, y_train, dataset_config=None, discriminator_config=None, gan_general_config=None,
                            gan_config=None, path_config=None):
    misc = Misc()
    print(misc.log_msg('WARNING', "----------------- Training AC-GAN ------------------"))
    X_train_segment = X_train
    X_train_segment = X_train_segment.reshape((X_train_segment.shape[0], X_train_segment.shape[1], 1))
    y_train_segment = y_train['FLOOR'].values  # 3-Floor, 4-Building

    # create the discriminator
    d_model = define_discriminator(X_train=X_train_segment, y_train=y_train_segment,
                                   discriminator_config=discriminator_config)
    # create the generator
    g_model = define_generator(X_train=X_train_segment, y_train=y_train_segment)
    # create the gan
    gan_model = define_gan(g_model, d_model, gan_config=gan_config)
    # train model
    train(X_train_segment, y_train_segment, 0, g_model, d_model, gan_model, dataset_config=dataset_config,
          gan_general_config=gan_general_config, path_config=path_config)
