from Classes_ML.AE import *
from Classes_ML.VAE import *
from Classes_ML.AAE import *
from Classes_ML.Interfaces import Factory
from Classes_ML.Prior import *

class ModelFactoryDL(Factory):

    def __init__(self, hyper_model):

        self.hyper_model = hyper_model

    def create(self) -> IModelDL:

        if self.hyper_model['model_name'] == "AE":
            scopeAE = ScopeAE(hyper_model=self.hyper_model)
            model = AutoEncoder(self.hyper_model, scope=scopeAE)
            model.create()
        elif self.hyper_model['model_name'] == "VAE":
            scopeVAE = ScopeVAE(hyper_model=self.hyper_model)
            model = VariationalAutoEncoder(self.hyper_model, scope=scopeVAE)
            model.create()
        elif self.hyper_model['model_name'] == "UAAE":
            scopeUAAE = ScopeUAAE(hyper_model=self.hyper_model)
            model = UAAE(self.hyper_model, scope=scopeUAAE)
            model.create()
        elif self.hyper_model['model_name'] == "SSAAE":
            scopeSSAAE = ScopeSSAAE(hyper_model=self.hyper_model)
            model = SSAAE(self.hyper_model, scope=scopeSSAAE)
            model.create()
        else:
            model = None

        return model

class Model_DL(IModel):

    def __init__(self, hyper_model: dict):

        """
        Create Deep Learning model corresponding to the input dictionary hyper_model hyperparameters

        Input
        -------
        hyper_model : dict
            2D array of data that is dimensionally reduced dim(num_train_data, 2)

        Returns
        -------
        None
        """

        self.hyper_model = hyper_model
        self.model_name = self.hyper_model['model_name']
        self.max_epoch = self.hyper_model['max_epoch']
        self.batch_size = self.hyper_model['batch_size']
        self.metric_function = self.hyper_model['metric_function']

        self.create_model()

    def create_model(self):

        self.modelFactory = ModelFactoryDL(hyper_model=self.hyper_model)
        self.model = self.modelFactory.create()

    def fit(self, X_fit, y_fit, X_transform, y_transform):

        self.loss_dict = self.model.fit(x=X_fit, y=y_fit, epochs=self.max_epoch, batch_size=self.batch_size,
                                                    verbose=1, validation_data=(X_transform, y_transform), shuffle=True)

        z_train = self.model.encoder_model.predict(X_fit)

        return z_train

    def transform(self, X_transform, y_transform):

        z_valid = self.model.encoder_model.predict(X_transform)

        return z_valid

    def transform_models_dict(self, X_transform, y_transform, K):

        z_test = self.models_dict[K]["Encoder"].predict(X_transform)

        return z_test

    def save_model(self, K, path_model=""):

        self.model.save_model(K=K, path_model=path_model)

        # name = str(self.model_name) + "_K_" + str(K) + ".pkl"
        #
        # self.model.encoder_model.save(path_model + "Encoder_" + name)
        # self.model.decoder_model.save(path_model + "Decoder_" + name)
        # self.model.autoencoder_model.save(path_model + "AutoEncoder_" + name)
        #
        # self.model.encoder_model.save_weights(path_model + "Encoder_" + name + ".h5")
        # self.model.decoder_model.save_weights(path_model + "Decoder_" + name + ".h5")
        # self.model.autoencoder_model.save_weights(path_model + "AutoEncoder_" + name + ".h5")
        #
        # print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "Encoder_" + name)
        # print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "Decoder_" + name)
        # print('Model ' + self.hyper_model['model_name'] + " saved at " + path_model + "AutoEncoder_" + name)

    def load_model(self, n_K_fold, path_model=""):

        self.models_dict = self.model.load_model(n_K_fold=n_K_fold, path_model=path_model)

        # self.models_dict = {}
        #
        # for K in range(0, n_K_fold):
        #
        #     self.models_dict[K] = {}
        #
        #     name = str(self.model_name) + "_K_" + str(K) + ".pkl"
        #
        #     self.model.encoder_model.load_weights(path_model + "Encoder_" + name + ".h5")
        #     self.model.decoder_model.load_weights(path_model + "Decoder_" + name + ".h5")
        #     self.model.autoencoder_model.load_weights(path_model + "AutoEncoder_" + name + ".h5")
        #
        #     self.models_dict[K]["Encoder"] = self.model.encoder_model
        #     self.models_dict[K]["Decoder"] = self.model.decoder_model
        #     self.models_dict[K]["AutoEncoder"] = self.model.autoencoder_model
        #
        #     print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "Encoder_" + name)
        #     print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "Decoder_" + name)
        #     print('Model ' + self.hyper_model['model_name'] + " loaded at " + path_model + "AutoEncoder_" + name)

    def save_best_nn(self, best_nn, K, path_model=""):

        name = str(self.model_name) + "_K_" + str(K) + "_best_nn.npy"
        np.save(path_model + name, best_nn)

    def load_best_nn(self, n_K_fold, path_model=""):

        self.best_nn_dict = {}

        for K in range(0, n_K_fold):

            # Load best nn for unsupervised
            name = str(self.model_name) + "_K_" + str(K) + "_best_nn.npy"
            self.best_nn_dict[K] = np.load(path_model + name)

    def show_config(self):

        df = pd.DataFrame.from_dict([self.hyper_model])
        print(tabulate(df, headers='keys', tablefmt='psql'))

    def fit_transform(self, X_fit, y_fit, X_transform, y_transform):

        # Nothing here for DL
        pass

    def fit_transform_models_dict(self, X_fit, y_fit, X_transform, y_transform, K):

        # Nothing here for DL
        pass
