# example of training an conditional gan on the fashion mnist dataset
import numpy as np
from numpy import expand_dims
import os
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout

from miscellaneous.misc import Misc


class CGAN:
    def __init__(self, X_train, y_train, dataset_config, general_config, discriminator_config, generator_config,
                 gan_config, path_config):
        self.n_classes = np.size(np.unique(y_train[:, 2]))
        self.features = np.shape(X_train)[1]
        self.latent_dim = np.shape(X_train)[1]
        self.X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.y_train = y_train[:, 2]
        self.path_config = path_config
        self.gan_config = gan_config
        self.discriminator_config = discriminator_config
        self.generator_config = generator_config
        self.general_config = general_config
        self.dataset_config = dataset_config
        self.misc = Misc()

    def define_discriminator(self):
        # label input
        in_shape = (self.features, 1)
        
        in_label = Input(shape=(1,))
        # embedding for categorical input
        l_input = Embedding(self.n_classes, 50)(in_label)
        n_nodes = in_shape[0] * in_shape[1]
        l_input = Dense(n_nodes)(l_input)
        l_input = Reshape((in_shape[0], 1))(l_input)
        in_dataset = Input(shape=in_shape)
        merge = Concatenate()([in_dataset, l_input])
        disc_model = Conv1D(filters=32, kernel_size=4)(merge)
        disc_model = LeakyReLU(alpha=0.2)(disc_model)
        # disc_model = Dropout(0.5)(disc_model)
        disc_model = Conv1D(64, kernel_size=4)(disc_model)
        disc_model = LeakyReLU(alpha=0.2)(disc_model)
        # disc_model = Dropout(0.5)(disc_model)
        disc_model = Conv1D(128, kernel_size=4)(disc_model)
        disc_model = LeakyReLU(alpha=0.2)(disc_model)
        # disc_model = Dropout(0.5)(disc_model)
        disc_model = Conv1D(256, kernel_size=4)(disc_model)
        disc_model = LeakyReLU(alpha=0.2)(disc_model)
        disc_model = Conv1D(512, kernel_size=4)(disc_model)
        disc_model = LeakyReLU(alpha=0.2)(disc_model)
        # disc_model = Dropout(0.5)(disc_model)
        disc_model = Flatten()(disc_model)
        # dropout
        disc_model = Dropout(0.4)(disc_model)

        output = Dense(1, activation='sigmoid')(disc_model)
        model = Model([in_dataset, in_label], output)
        optimizer = self.misc.optimizer(self.discriminator_config['optimizer'], self.discriminator_config['lr'])
        model.compile(loss=self.discriminator_config['loss'], optimizer=optimizer, metrics=['accuracy'])
        return model

    def define_generator(self):
        in_label = Input(shape=(1,))
        l_input = Embedding(self.n_classes, 50)(in_label)
        n_nodes = 1 * self.features
        l_input = Dense(n_nodes)(l_input)
        l_input = Reshape((self.features, 1))(l_input)
        in_lat = Input(shape=(self.latent_dim,))
        n_nodes = 1 * self.features
        generator = Dense(n_nodes)(in_lat)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = Reshape((self.features, 1))(generator)
        merge = Concatenate()([generator, l_input])
        generator = Conv1D(32, 4, padding='same')(merge)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = Conv1D(64, 4, padding='same')(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        # generator = Dropout(0.5)(generator)
        generator = Conv1D(128, 4, padding='same')(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        # generator = Dropout(0.5)(generator)
        # output
        out_layer = Conv1D(1, 1, activation='elu', padding='same')(generator)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model

    def define_gan(self):
        d_model = self.define_discriminator()
        d_model.trainable = False
        g_model = self.define_generator()
        gen_noise, gen_label = g_model.input
        gen_output = g_model.output
        gan_output = d_model([gen_output, gen_label])
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        optimizer = self.misc.optimizer(self.gan_config['optimizer'], self.gan_config['lr'])
        model.compile(loss=self.gan_config['loss'], optimizer=optimizer)
        return model

    def generate_latent_points(self):
        x_input = randn(self.latent_dim * self.general_config['num_fake_samples'])
        z_input = x_input.reshape(self.general_config['num_fake_samples'], self.latent_dim)
        labels = randint(0, self.n_classes, self.general_config['num_fake_samples'])
        return [z_input, labels]

    def generate_fake_samples(self, generator):
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points()
        images = generator.predict([z_input, labels_input])
        y = zeros((self.general_config['num_fake_samples'], 1))
        return [images, labels_input], y

    def train(self):
        main_path_save = os.path.join(self.path_config['saved_model'], self.dataset_config['name'])
        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)

        if self.general_config['train'] == 'True':
            g_model = self.define_generator()
            d_model = self.define_discriminator()
            gan_model = self.define_gan()
            bat_per_epo = int(self.X_train.shape[0] / self.general_config['batch_size'])
            # manually enumerate epochs
            for i in range(self.general_config['epochs']):
                # enumerate batches over the training set
                for j in range(bat_per_epo):
                    X_real = self.X_train
                    labels_real = self.y_train
                    y_real = np.ones(np.size(labels_real))
                    # update discriminator model weights
                    d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                    # Generate fake samples
                    [X_fake, labels], y_fake = self.generate_fake_samples(g_model)
                    # update discriminator model weights
                    d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
                    # evaluate the model
                    _, acc_real = d_model.evaluate([X_real, labels_real], y_real, verbose=0)
                    _, acc_fake = d_model.evaluate([X_fake, labels], y_fake, verbose=0)
                    # prepare points in latent space as input for the generator
                    [z_input, labels_input] = self.generate_latent_points()
                    # create inverted labels for the fake samples
                    y_gan = ones((self.general_config['num_fake_samples'], 1))
                    # update the generator via the discriminator's error
                    g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
                    # summarize loss on this batch
                    print('>Epoch: %d, Batch epoch %d/%d, Discriminator real - Loss %.4f, '
                          'Discriminator fake loss %.4f, GAN loss %.4f' %
                          (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
                    # print("Epoch {:.1f}, accuracy real {:.3f}, accuracy fake {:.3f}".format(i, acc_real, acc_fake))
            g_model.save(main_path_save + '/cgan_radio_map_generator.h5')
        # else:
        #    g_model = load_model(main_path_save + '/cgan_radio_map_generator.h5')
        return True

