# example of training an conditional gan on the fashion mnist dataset
import numpy as np
import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from collections import Counter
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
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

gpu_available = tf.test.is_gpu_available()

if gpu_available:
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


# define the standalone discriminator model
def define_discriminator(X_train=None, y_train=None, discriminator_config=None):
    # label input
    in_shape = (np.shape(X_train)[1], 1)
    n_classes = int(max(y_train) + 1)
    in_label = Input(shape=(1,))

    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], 1))(li)
    # Hasta aquí tengo el input (None, 520, 1)
    in_dataset = Input(shape=in_shape)
    merge = Concatenate()([in_dataset, li])
    fe = Conv1D(filters=32, kernel_size=4)(merge)
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
    model = Model([in_dataset, in_label], output)
    opt = Adam(learning_rate=0.0002)
    misc = Misc()
    optimizer = misc.optimizer(discriminator_config['optimizer'], discriminator_config['lr'])
    model.compile(loss=discriminator_config['loss'], optimizer=optimizer, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(X_train=None, y_train=None):
    # label input
    latent_dim = np.shape(X_train)[1]
    features = np.shape(X_train)[1]
    n_classes = int(max(y_train) + 1)

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
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    misc = Misc()
    optimizer = misc.optimizer(gan_config['optimizer'], gan_config['lr'])
    model.compile(loss=gan_config['loss'], optimizer=optimizer)
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
          path_config=None, building=None, algorithm=None, method=None):
    if gan_general_config['train'] == 'True':
        n_samples = gan_general_config['num_fake_samples']
        n_epochs = gan_general_config['epochs']
        latent_dim = np.shape(X_train)[1]
        n_batch = gan_general_config['batch_size']
        half_batch = int(n_batch / 2)
        bat_per_epo = int(np.shape(X_train)[0] / n_batch)
        misc = Misc()

        sub_path = "epochs_" + str(gan_general_config['epochs']) + '_bs_' + str(gan_general_config['batch_size'])
        main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'], algorithm, method, sub_path)

        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)
        iteration = str(iteration)

        if building is None:
            building = 0
        print(misc.log_msg('WARNING', "---- Building " + str(building) + "- Iteration " + iteration.zfill(2) + " ----"))

        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, labels_real, y_real = data_selection(X_train=X_train, y_train=y_train,
                                                             num_samples=half_batch)
                # X_real = X_train
                # labels_real = y_train
                n_classes = max(y_train) + 1
                # y_real = np.ones(np.size(labels_real))

                # update discriminator model weights
                d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                # Generate fake samples
                [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_classes)
                # update discriminator model weights
                d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
                # evaluate the model
                _, acc_real = d_model.evaluate([X_real, labels_real], y_real, verbose=0)
                _, acc_fake = d_model.evaluate([X_fake, labels], y_fake, verbose=0)
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, n_classes)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>Epoch: %d, Batch epoch %d/%d, Discriminator real - Loss %.4f, '
                      'Discriminator fake loss %.4f, GAN loss %.4f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
                # print("Epoch {:.1f}, accuracy real {:.3f}, accuracy fake {:.3f}".format(i, acc_real, acc_fake))

        g_model.save(main_path_save + '/cgan_generator_building_' + str(gan_general_config['epochs']) + '_' +
                     str(gan_general_config['batch_size']) + '_building_' + str(building) + '_iteration_' +
                     iteration.zfill(2) + '.h5')


def train_imbalance_classes_building(X_train, y_train, dataset_config=None, discriminator_config=None, gan_general_config=None,
                            gan_config=None, path_config=None, algorithm=None, method=None):
    misc = Misc()
    print(misc.log_msg('WARNING', "----------------- Training GAN ------------------"))
    classes_val = Counter(y_train.values[:, 4])  # 3- Floor, 4- Building
    class_val_sort = classes_val.most_common()
    ratio = []
    index = []
    samples = []

    for i in range(0, len(class_val_sort)):
        index.append(class_val_sort[i][0])
        samples.append(class_val_sort[i][1])
        ratio.append(class_val_sort[0][1] / class_val_sort[i][1])

    # Get all the classes with a ratio grater than or equal to 1
    list_candidates = [idx for idx, element in enumerate(ratio) if element > 1]

    for candidate, idx_class in enumerate(list_candidates):
        index_samples = y_train[y_train['BUILDINGID'] == int(index[idx_class])].index
        X_train_selected = X_train[index_samples, :]
        y_train_selected = y_train[y_train['BUILDINGID'] == int(index[idx_class])].values
        X_train_segment = X_train_selected
        X_train_segment = X_train_segment.reshape((X_train_segment.shape[0], X_train_segment.shape[1], 1))
        y_train_segment = y_train_selected[:, 3]  # 3-Floor, 4-Building

        # create the discriminator
        d_model = define_discriminator(X_train=X_train_segment, y_train=y_train_segment,
                                       discriminator_config=discriminator_config)
        # create the generator
        g_model = define_generator(X_train=X_train_segment, y_train=y_train_segment)
        # create the gan
        gan_model = define_gan(g_model, d_model, gan_config=gan_config)
        # train model
        train(X_train_segment, y_train_segment, 0, g_model, d_model, gan_model, dataset_config=dataset_config,
              gan_general_config=gan_general_config, path_config=path_config, building=int(index[idx_class]),
              algorithm=algorithm, method=method)
