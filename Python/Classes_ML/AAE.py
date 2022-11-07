import math
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import optimizers, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Input, Conv2D, BatchNormalization, Reshape, AveragePooling2D, MaxPooling2D, concatenate
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

from Classes_ML.Interfaces import IModelDL, IModel, IScope
from Classes_ML.Prior import PriorDistribution
from Classes_ML.Loss import LossFunctions
from Classes_ML.LayersUtilities import LayersUtilities

class ScopeUAAE(IScope):

    optimizer_autoencoder = None
    optimizer_discriminator = None
    optimizer_generator = None

    autoencoder_metric = None
    discriminator_metric = None
    generator_metric = None

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model

        self.optimizer_name = hyper_model['optimizer_name']
        self.batch_size = hyper_model['batch_size']
        self.max_epoch = hyper_model['max_epoch']

        self.learning_rate_schedule = self.hyper_model["learning_rate_schedule"]
        self.learning_rate_start = self.learning_rate_schedule["learning_rate_start"]

        self.metric_function = self.hyper_model["metric_function"]
        self.loss_function = self.hyper_model["loss_function"]

    def create(self, clear_session=True):

        # Create graph and session
        if clear_session:
            backend.clear_session()

        with tf.name_scope("loss"):

            self.autoencoder_loss = LossFunctions.get_mse()
            self.discriminator_loss = LossFunctions.get_binary_cross_entropy()
            self.generator_loss = LossFunctions.get_binary_cross_entropy()

        with tf.name_scope("Metrics"):

            self.autoencoder_metric = LossFunctions.rmse
            self.discriminator_metric = LossFunctions.rmse
            self.generator_metric = LossFunctions.rmse

        with tf.name_scope("Optimizer"):

            if self.optimizer_name == 'adam':
                self.optimizer_autoencoder = optimizers.Adam(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adam(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adam(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'rmsprop':
                self.optimizer_autoencoder = optimizers.RMSprop(learning_rate=self.learning_rate_start, decay=0.0002)
                self.optimizer_discriminator = optimizers.RMSprop(learning_rate=self.learning_rate_start, decay=0.0002)
                self.optimizer_generator = optimizers.RMSprop(learning_rate=self.learning_rate_start, decay=0.0002)
            elif self.optimizer_name == 'adadelta':
                self.optimizer_autoencoder = optimizers.Adadelta(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adadelta(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adadelta(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'adagrad':
                self.optimizer_autoencoder = optimizers.Adagrad(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adagrad(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adagrad(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'sgd':
                self.optimizer_autoencoder = optimizers.SGD(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.SGD(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.SGD(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'adamax':
                self.optimizer_autoencoder = optimizers.Adamax(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adamax(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adamax(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'nadam':
                self.optimizer_autoencoder = optimizers.Nadam(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Nadam(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Nadam(learning_rate=self.learning_rate_start)

class ScopeSSAAE(IScope):

    optimizer_autoencoder = None
    optimizer_discriminator = None
    optimizer_generator = None

    autoencoder_metric = None
    discriminator_metric = None
    generator_metric = None

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model

        self.optimizer_name = hyper_model['optimizer_name']
        self.batch_size = hyper_model['batch_size']
        self.max_epoch = hyper_model['max_epoch']

        self.learning_rate_schedule = self.hyper_model["learning_rate_schedule"]
        self.learning_rate_start = self.learning_rate_schedule["learning_rate_start"]

        self.metric_function = self.hyper_model["metric_function"]
        self.loss_function = self.hyper_model["loss_function"]

    def create(self, clear_session=True):

        # Create graph and session
        if clear_session:
            backend.clear_session()

        with tf.name_scope("loss"):
            self.autoencoder_loss = LossFunctions.get_mse()
            self.discriminator_loss = LossFunctions.get_binary_cross_entropy()
            self.generator_loss = LossFunctions.get_binary_cross_entropy()

        with tf.name_scope("Metrics"):
            self.autoencoder_metric = LossFunctions.rmse
            self.discriminator_metric = LossFunctions.rmse
            self.generator_metric = LossFunctions.rmse

        with tf.name_scope("Optimizer"):

            if self.optimizer_name == 'adam':
                self.optimizer_autoencoder = optimizers.Adam(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adam(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adam(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'rmsprop':
                self.optimizer_autoencoder = optimizers.RMSprop(learning_rate=self.learning_rate_start, decay=0.0002)
                self.optimizer_discriminator = optimizers.RMSprop(learning_rate=self.learning_rate_start, decay=0.0002)
                self.optimizer_generator = optimizers.RMSprop(learning_rate=self.learning_rate_start, decay=0.0002)
            elif self.optimizer_name == 'adadelta':
                self.optimizer_autoencoder = optimizers.Adadelta(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adadelta(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adadelta(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'adagrad':
                self.optimizer_autoencoder = optimizers.Adagrad(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adagrad(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adagrad(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'sgd':
                self.optimizer_autoencoder = optimizers.SGD(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.SGD(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.SGD(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'adamax':
                self.optimizer_autoencoder = optimizers.Adamax(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Adamax(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Adamax(learning_rate=self.learning_rate_start)
            elif self.optimizer_name == 'nadam':
                self.optimizer_autoencoder = optimizers.Nadam(learning_rate=self.learning_rate_start)
                self.optimizer_discriminator = optimizers.Nadam(learning_rate=self.learning_rate_start)
                self.optimizer_generator = optimizers.Nadam(learning_rate=self.learning_rate_start)

class UAAE(IModelDL):

    encoder_model = None
    decoder_model = None
    autoencoder_model = None
    discriminator_model = None
    generator_model = None

    def __init__(self, hyper_model, scope: ScopeUAAE):

        self.hyper_model = hyper_model
        self.scope = scope

        self.input_dim = self.hyper_model["input_dim"]
        self.latent_dim = self.hyper_model["latent_dim"]
        self.encoder_units = self.hyper_model["encoder_units"]

        self.decoder_units = self.hyper_model["encoder_units"][::-1]

        self.discriminator_units = self.hyper_model["discriminator_units"]

        self.encoder_activation = self.hyper_model["encoder_activation"]
        self.decoder_activation = self.hyper_model["encoder_activation"]
        self.discriminator_activation = self.hyper_model["discriminator_activation"]

        self.dropout_rate = self.hyper_model["dropout_rate"]
        self.use_batch_normalization = self.hyper_model["use_batch_normalization"]

        self.kernel_regularizer = LayersUtilities.regularization(name=self.hyper_model["kernel_regularizer_name"])
        self.kernel_initializer = LayersUtilities.kernel_initializer(kernel_initializer_info=self.hyper_model["kernel_initializer_info"])

        self.prior = self.hyper_model["prior"]

        self.model_name = self.hyper_model["model_name"]

    def create(self):

        # Scope
        self.scope.create()

        # Input AE
        self.x_input = Input(shape=(self.input_dim[0],), name='x')

        # Encoder
        self.z = self.create_encoder(x=self.x_input)
        self.encoder_model = Model(inputs=[self.x_input], outputs=[self.z], name="Encoder")

        # Decoder
        self.latent_inputs = Input(shape=(self.z.get_shape().as_list()[-1],), name='latent_inputs')
        self.x_hat = self.create_decoder(z=self.latent_inputs)
        self.decoder_model = Model(inputs=[self.latent_inputs], outputs=[self.x_hat], name="Decoder")

        # Autoencoder
        decoder_output = self.decoder_model(self.encoder_model(self.x_input))
        self.autoencoder_model = Model(inputs=[self.x_input], outputs=[decoder_output], name="AutoEncoder")

        # Discriminator
        self.y_disc = self.create_discriminator(z=self.latent_inputs)
        self.discriminator_model = Model(inputs=[self.latent_inputs], outputs=[self.y_disc], name="Discriminator")

        # Generator
        generator_output = self.discriminator_model(self.encoder_model(self.x_input))
        self.generator_model = Model(inputs=[self.x_input], outputs=[generator_output], name="Generator")

        # Compile
        self.autoencoder_model.compile(loss=[self.scope.autoencoder_loss], optimizer=self.scope.optimizer_autoencoder, metrics=[self.scope.autoencoder_metric])
        self.discriminator_model.compile(loss=[self.scope.discriminator_loss], optimizer=self.scope.optimizer_discriminator, metrics=[self.scope.discriminator_metric])

        for layer in self.discriminator_model.layers:
            layer.trainable = False

        self.generator_model.compile(loss=[self.scope.generator_loss], optimizer=self.scope.optimizer_generator, metrics=[self.scope.generator_metric])

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoencoder_model.summary()
        self.discriminator_model.summary()
        self.generator_model.summary()

        self.count_params = self.autoencoder_model.count_params()

    def create_encoder(self, x):

        l = x

        for i in range(0, len(self.encoder_units)):

            l = Dense(units=self.encoder_units[i], activation=self.encoder_activation,
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='encoder_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='encoder_batch_norm_' + str(i))(l)

            if self.dropout_rate > 0.0:
                l = Dropout(rate=self.dropout_rate, name='encoder_dropout_' + str(i))(l)

        z = Dense(units=self.latent_dim[0], activation="linear", name='z')(l)

        return z

    def create_decoder(self, z):

        l = z

        for i in range(0, len(self.decoder_units)):

            l = Dense(units=self.decoder_units[i], activation=self.decoder_activation,
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='decoder_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='decoder_batch_norm_' + str(i))(l)

            if self.dropout_rate > 0.0:
                l = Dropout(rate=self.dropout_rate, name='decoder_dropout_' + str(i))(l)

        x_hat = Dense(units=self.input_dim[0], activation="linear", name='x_hat')(l)

        return x_hat

    def create_discriminator(self, z):

        l = z

        for i in range(0, len(self.discriminator_units)):

            l = Dense(units=self.discriminator_units[i], activation=self.discriminator_activation,
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='discriminator_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='discriminator_batch_norm_' + str(i))(l)

            if self.dropout_rate > 0.0:
                l = Dropout(rate=self.dropout_rate, name='discriminator_dropout_' + str(i))(l)

        y = Dense(units=1, activation="sigmoid", name='y')(l)

        return y

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, validation_data=None, shuffle=True):

        n_batch = x.shape[0] // batch_size

        for i in range(epochs):

            tic = time.time()

            for n in range(n_batch):

                # Take random sample
                indices = np.random.randint(0, x.shape[0], batch_size)
                x_train = x[indices]

                loss_autoencoder = self.autoencoder_model.train_on_batch(x_train, x_train)

                # Fake latent space
                z_fake = self.encoder_model.predict(x_train)
                y_fake = np.zeros((batch_size, 1))

                z_real = PriorDistribution.generate_z_un_N(batch_size=batch_size, n_labels=self.prior["size"], x_std=self.prior["x_std"], y_std=self.prior["y_std"], shift=self.prior["shift"])
                y_real = np.ones((batch_size, 1))

                z = np.vstack([z_fake, z_real])
                y = np.vstack([y_fake, y_real])

                loss_discriminator = self.discriminator_model.train_on_batch(z, y)

                y_real = np.ones((batch_size, 1))
                loss_generator = self.generator_model.train_on_batch(x_train, y_real)

            toc = time.time()
            print("Epoch " + str(i) + " - Time: " + str(toc - tic) + " s")
            print("Epoch " + str(i) + " - [AE loss: %f, AE rmse: %.2f]" % (loss_autoencoder[0], loss_autoencoder[0]))
            print("Epoch " + str(i) + " - [D loss: %f, D rmse: %.2f]" % (loss_discriminator[0], loss_discriminator[1]))
            print("Epoch " + str(i) + " - [G loss: %f, G rmse: %.2f]" % (loss_generator[0], loss_generator[0]))

        if math.isnan(loss_discriminator[0]):
            loss_dict = dict(AE_loss=999, AE_metric=999,
                             D_loss=999, D_metric=999,
                             G_loss=999, G_metric=999)  # Use 999 to badly ranked the model
        else:
            loss_dict = dict(AE_loss=loss_autoencoder[0], AE_metric=loss_autoencoder[0],
                             D_loss=loss_discriminator[0], D_metric=loss_discriminator[1],
                             G_loss=loss_generator[0], G_metric=loss_generator[0])

        return loss_dict

    def save_model(self, K, path_model=""):

        name = str(self.model_name) + "_K_" + str(K) + ".pkl"

        self.encoder_model.save(path_model + "Encoder_" + name)
        self.decoder_model.save(path_model + "Decoder_" + name)
        self.autoencoder_model.save(path_model + "AutoEncoder_" + name)

        self.encoder_model.save_weights(path_model + "Encoder_" + name + ".h5")
        self.decoder_model.save_weights(path_model + "Decoder_" + name + ".h5")
        self.autoencoder_model.save_weights(path_model + "AutoEncoder_" + name + ".h5")

        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "Encoder_" + name)
        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "Decoder_" + name)
        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "AutoEncoder_" + name)

    def load_model(self, n_K_fold, path_model=""):

        self.models_dict = {}

        for K in range(0, n_K_fold):
            self.models_dict[K] = {}

            name = str(self.model_name) + "_K_" + str(K) + ".pkl"

            self.encoder_model.load_weights(path_model + "Encoder_" + name + ".h5")
            self.decoder_model.load_weights(path_model + "Decoder_" + name + ".h5")
            self.autoencoder_model.load_weights(path_model + "AutoEncoder_" + name + ".h5")

            self.models_dict[K]["Encoder"] = self.encoder_model
            self.models_dict[K]["Decoder"] = self.decoder_model
            self.models_dict[K]["AutoEncoder"] = self.autoencoder_model

            print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "Encoder_" + name)
            print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "Decoder_" + name)
            print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "AutoEncoder_" + name)

        return self.models_dict

class SSAAE(IModelDL):

    encoder_model = None
    decoder_model = None
    autoencoder_model = None
    discriminator_model = None
    generator_model = None

    def __init__(self, hyper_model, scope: ScopeSSAAE):

        self.hyper_model = hyper_model
        self.scope = scope

        self.input_dim = self.hyper_model["input_dim"]
        self.latent_dim = self.hyper_model["latent_dim"]
        self.encoder_units = self.hyper_model["encoder_units"]

        self.decoder_units = self.hyper_model["encoder_units"][::-1]

        self.discriminator_units = self.hyper_model["discriminator_units"]

        self.encoder_activation = self.hyper_model["encoder_activation"]
        self.decoder_activation = self.hyper_model["encoder_activation"]
        self.discriminator_activation = self.hyper_model["discriminator_activation"]

        self.kernel_regularizer = LayersUtilities.regularization(name=self.hyper_model["kernel_regularizer_name"])
        self.kernel_initializer = LayersUtilities.kernel_initializer(kernel_initializer_info=self.hyper_model["kernel_initializer_info"])

        self.prior = self.hyper_model["prior"]

        self.dropout_rate = self.hyper_model["dropout_rate"]
        self.use_batch_normalization = self.hyper_model["use_batch_normalization"]

        self.model_name = self.hyper_model["model_name"]

    def create(self):

        # Scope
        self.scope.create()

        # Input AE
        self.x_input = Input(shape=(self.input_dim[0],), name='x')

        # Encoder
        self.z = self.create_encoder(x=self.x_input)
        self.encoder_model = Model(inputs=[self.x_input], outputs=[self.z], name="Encoder")

        # Decoder
        self.latent_inputs = Input(shape=(self.z.get_shape().as_list()[-1],), name='latent_inputs')
        self.x_hat = self.create_decoder(z=self.latent_inputs)
        self.decoder_model = Model(inputs=[self.latent_inputs], outputs=[self.x_hat], name="Decoder")

        # Autoencoder
        decoder_output = self.decoder_model(self.encoder_model(self.x_input))
        self.autoencoder_model = Model(inputs=[self.x_input], outputs=[decoder_output], name="AutoEncoder")

        # Input layer for label in discriminator
        self.label_inputs_disc = Input(shape=(self.prior["size"],), name="label_inputs_disc")

        # Concatenate latent space and label
        self.latent_label_inputs = concatenate([self.latent_inputs, self.label_inputs_disc], name="latent_label_inputs")

        # Discriminator
        self.y_disc = self.create_discriminator(z=self.latent_label_inputs)
        self.discriminator_model = Model(inputs=[self.latent_inputs, self.label_inputs_disc], outputs=[self.y_disc], name="Discriminator")

        # Input layer for label in generator
        self.label_inputs_generator = Input(shape=(self.prior["size"],), name="label_inputs_generator")

        # Generator
        generator_output = self.discriminator_model([self.encoder_model(self.x_input), self.label_inputs_generator])
        self.generator_model = Model(inputs=[self.x_input, self.label_inputs_generator], outputs=[generator_output], name="Generator")

        # Compile
        self.autoencoder_model.compile(loss=[self.scope.autoencoder_loss], optimizer=self.scope.optimizer_autoencoder, metrics=[self.scope.autoencoder_metric])
        self.discriminator_model.compile(loss=[self.scope.discriminator_loss], optimizer=self.scope.optimizer_discriminator, metrics=[self.scope.discriminator_metric])

        for layer in self.discriminator_model.layers:
            layer.trainable = False

        self.generator_model.compile(loss=[self.scope.generator_loss], optimizer=self.scope.optimizer_generator, metrics=[self.scope.generator_metric])

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoencoder_model.summary()
        self.discriminator_model.summary()
        self.generator_model.summary()

        self.count_params = self.autoencoder_model.count_params()

    def create_encoder(self, x):

        l = x

        for i in range(0, len(self.encoder_units)):

            l = Dense(units=self.encoder_units[i], activation=self.encoder_activation,
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='encoder_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='encoder_batch_norm_' + str(i))(l)

            if self.dropout_rate > 0.0:
                l = Dropout(rate=self.dropout_rate, name='encoder_dropout_' + str(i))(l)

        z = Dense(units=self.latent_dim[0], activation="linear",
                  kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                  name='z')(l)

        return z

    def create_decoder(self, z):

        l = z

        for i in range(0, len(self.decoder_units)):

            l = Dense(units=self.decoder_units[i], activation=self.decoder_activation,
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='decoder_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='decoder_batch_norm_' + str(i))(l)

            if self.dropout_rate > 0.0:
                l = Dropout(rate=self.dropout_rate, name='decoder_dropout_' + str(i))(l)

        x_hat = Dense(units=self.input_dim[0], activation="linear",
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='x_hat')(l)

        return x_hat

    def create_discriminator(self, z):

        l = z

        for i in range(0, len(self.discriminator_units)):

            l = Dense(units=self.discriminator_units[i], activation=self.discriminator_activation,
                      kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                      name='discriminator_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='discriminator_batch_norm_' + str(i))(l)

            if self.dropout_rate > 0.0:
                l = Dropout(rate=self.dropout_rate, name='discriminator_dropout_' + str(i))(l)

        y = Dense(units=1, activation="sigmoid",
                  kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.kernel_initializer, bias_initializer=self.kernel_initializer,
                  name='y')(l)

        return y

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, validation_data=None, shuffle=True):

        n_batch = x.shape[0] // batch_size

        for i in range(epochs):

            tic = time.time()

            for n in range(n_batch):

                # Take random sample
                indices = np.random.randint(0, x.shape[0], batch_size)
                x_train = x[indices]
                y_train = y[indices]

                loss_autoencoder = self.autoencoder_model.train_on_batch(x_train, x_train)

                # Fake latent space
                z_fake = self.encoder_model.predict(x_train)
                y_fake = np.zeros((batch_size, 1))

                z_real, labels = PriorDistribution.generate_z_ss_N(batch_size=batch_size, y=y_train, n_labels=self.prior["size"],
                                                                   x_std=self.prior["x_std"], y_std=self.prior["y_std"], shift=self.prior["shift"])
                y_real = np.ones((batch_size, 1))

                z_fake_real = np.vstack([z_fake, z_real])
                y_fake_real = np.vstack([y_fake, y_real])
                label_d = np.vstack([labels, labels])

                loss_discriminator = self.discriminator_model.train_on_batch([z_fake_real, label_d], y_fake_real)

                y_real = np.ones((batch_size, 1))
                loss_generator = self.generator_model.train_on_batch([x_train, labels], y_real)

            toc = time.time()
            print("Epoch " + str(i) + " - Time: " + str(toc - tic) + " s")
            print("Epoch " + str(i) + " - [AE loss: %f, AE rmse: %f]" % (loss_autoencoder[0], loss_autoencoder[0]))
            print("Epoch " + str(i) + " - [D loss: %f, D rmse: %f]" % (loss_discriminator[0], loss_discriminator[1]))
            print("Epoch " + str(i) + " - [G loss: %f, G rmse: %f]" % (loss_generator[0], loss_generator[0]))

        if math.isnan(loss_discriminator[0]):
            loss_dict = dict(AE_loss=999, AE_metric=999,
                             D_loss=999, D_metric=999,
                             G_loss=999, G_metric=999)  # Use 999 to badly ranked the model
        else:
            loss_dict = dict(AE_loss=loss_autoencoder[0], AE_metric=loss_autoencoder[0],
                             D_loss=loss_discriminator[0], D_metric=loss_discriminator[1],
                             G_loss=loss_generator[0], G_metric=loss_generator[0])

        return loss_dict

    def save_model(self, K, path_model=""):

        name = str(self.model_name) + "_K_" + str(K) + ".pkl"

        self.encoder_model.save(path_model + "Encoder_" + name)
        self.decoder_model.save(path_model + "Decoder_" + name)
        self.autoencoder_model.save(path_model + "AutoEncoder_" + name)

        self.encoder_model.save_weights(path_model + "Encoder_" + name + ".h5")
        self.decoder_model.save_weights(path_model + "Decoder_" + name + ".h5")
        self.autoencoder_model.save_weights(path_model + "AutoEncoder_" + name + ".h5")

        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "Encoder_" + name)
        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "Decoder_" + name)
        print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "AutoEncoder_" + name)

    def load_model(self, n_K_fold, path_model=""):

        self.models_dict = {}

        for K in range(0, n_K_fold):
            self.models_dict[K] = {}

            name = str(self.model_name) + "_K_" + str(K) + ".pkl"

            self.encoder_model.load_weights(path_model + "Encoder_" + name + ".h5")
            self.decoder_model.load_weights(path_model + "Decoder_" + name + ".h5")
            self.autoencoder_model.load_weights(path_model + "AutoEncoder_" + name + ".h5")

            self.models_dict[K]["Encoder"] = self.encoder_model
            self.models_dict[K]["Decoder"] = self.decoder_model
            self.models_dict[K]["AutoEncoder"] = self.autoencoder_model

            print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "Encoder_" + name)
            print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "Decoder_" + name)
            print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "AutoEncoder_" + name)

        return self.models_dict
