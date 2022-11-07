import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Input, Conv2D, BatchNormalization, Reshape, AveragePooling2D, MaxPooling2D
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.losses import logcosh, CategoricalCrossentropy, Huber, binary_crossentropy
from tensorflow.keras.models import load_model
from Classes_ML.Loss import *

from Classes_ML.Interfaces import IModelDL, IModel, IScope

class ScopeAE(IScope):

    loss = None
    metrics_list = []
    optimizer = None

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

            if self.loss_function["name"] == "rmse":
                self.loss = LossFunctions.rmse

        with tf.name_scope("Metrics"):

            if self.metric_function["name"] == "rmse":
                self.metrics_list = [LossFunctions.rmse]

        with tf.name_scope("Optimizer"):

            if self.optimizer_name == 'adam':
                self.optimizer = optimizers.Adam(self.learning_rate_start, decay=0.0002, amsgrad=False)
            elif self.optimizer_name == 'rmsprop':
                self.optimizer = optimizers.RMSprop(self.learning_rate_start, decay=0.0002)
            elif self.optimizer_name == 'adadelta':
                self.optimizer = optimizers.Adadelta(self.learning_rate_start)
            elif self.optimizer_name == 'adagrad':
                self.optimizer = optimizers.Adagrad(self.learning_rate_start)
            elif self.optimizer_name == 'sgd':
                self.optimizer = optimizers.SGD(self.learning_rate_start, decay=0.0002)
            elif self.optimizer_name == 'adamax':
                self.optimizer = optimizers.Adamax(self.learning_rate_start)
            elif self.optimizer_name == 'nadam':
                self.optimizer = optimizers.Nadam(self.learning_rate_start)

class AutoEncoder(IModelDL):

    encoder_model = None
    decoder_model = None
    autoencoder_model = None

    def __init__(self, hyper_model, scope: IScope):

        self.hyper_model = hyper_model
        self.scope = scope

        self.input_dim = self.hyper_model["input_dim"]
        self.latent_dim = self.hyper_model["latent_dim"]
        self.encoder_units = self.hyper_model["encoder_units"]

        self.decoder_units = self.hyper_model["encoder_units"][::-1]

        self.encoder_activation = self.hyper_model["activation"]
        self.decoder_activation = self.hyper_model["activation"]

        self.dropout_rate = self.hyper_model["dropout_rate"]
        self.use_batch_normalization = self.hyper_model["use_batch_normalization"]

        self.model_name = self.hyper_model["model_name"]

    def create(self):

        # Scope
        self.scope.create()

        # Encoder
        self.x = Input(shape=(self.input_dim[0],))
        self.z = self.create_encoder(x=self.x)

        # Decoder
        self.z_input = Input(shape=(self.z.get_shape().as_list()[-1]), name='z')
        self.x_hat = self.create_decoder(z=self.z_input)

        # Create each model. The layers weight will be shared here since we are not recreating the layer. Here we just connect
        self.encoder_model = Model(inputs=[self.x], outputs=[self.z], name="Encoder")
        self.decoder_model = Model(inputs=[self.z_input], outputs=[self.x_hat], name="Decoder")
        self.autoencoder_model = Model(inputs=[self.x], outputs=[self.decoder_model(self.encoder_model(self.x))], name="AutoEncoder")

        # Compile
        self.autoencoder_model.compile(loss=self.scope.loss, optimizer=self.scope.optimizer, metrics=self.scope.metrics_list)

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.autoencoder_model.summary()

        self.count_params = self.autoencoder_model.count_params()

    def create_encoder(self, x):

        l = x

        for i in range(0, len(self.encoder_units)):
            l = Dense(units=self.encoder_units[i], activation=self.encoder_activation, name='encoder_' + str(i))(l)

            if self.use_batch_normalization:
                l = BatchNormalization(name='encoder_batch_norm_' + str(i))(l)

            l = Dropout(rate=self.dropout_rate, name='encoder_dropout_' + str(i))(l)

        z = Dense(units=self.latent_dim[0], activation="linear", name='z')(l)

        return z

    def create_decoder(self, z):

        l = z

        for i in range(0, len(self.decoder_units)):
            l = Dense(units=self.decoder_units[i], activation=self.decoder_activation, name='decoder_' + str(i))(l)
            if self.use_batch_normalization:
                l = BatchNormalization(name='decoder_batch_norm_' + str(i))(l)
            l = Dropout(rate=self.dropout_rate, name='decoder_dropout_' + str(i))(l)

        x_hat = Dense(units=self.input_dim[0], activation="linear", name='x_hat')(l)

        return x_hat

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, validation_data=None, shuffle=True):

        hist = self.autoencoder_model.fit(x=x, y=x, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                          validation_data=(validation_data[0], validation_data[0]), shuffle=shuffle)

        print("here")

        if math.isnan(hist.history["loss"][-1]):
            loss_dict = dict(AE_loss=999, AE_metric=999)  # Use 999 to badly ranked the model
        else:
            loss_dict = dict(AE_loss=hist.history["loss"][-1], AE_metric=hist.history["rmse"][-1])

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