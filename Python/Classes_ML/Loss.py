from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

class LossFunctions():

    @staticmethod
    def VAE_loss(z_mean, z_log_variance, loss_lambda):
        def vae_loss_reconstruction(y_true, y_pred):
            loss_reconstruction = LossFunctions.rmse(y_true, y_pred)

            return loss_reconstruction

        def vae_loss_kl(z_mean, z_log_variance, loss_lambda):
            kl_loss = -0.5 * K.sum(1.0 + z_log_variance - K.square(z_mean) - K.exp(z_log_variance), axis=1) * loss_lambda

            return kl_loss

        def vae_loss(y_true, y_pred, kl_loss_lambda=loss_lambda):
            loss_reconstruction = vae_loss_reconstruction(y_true, y_pred)
            loss_kl = vae_loss_kl(y_true, y_pred, kl_loss_lambda)

            loss = loss_reconstruction + loss_kl

            return loss

        return vae_loss

    @staticmethod
    def get_binary_cross_entropy():
        return BinaryCrossentropy(from_logits=False)

    @staticmethod
    def get_mse():
        return MeanSquaredError()

    @staticmethod
    def rmse(y_true, y_pred):
        rmse = K.sqrt(K.mean(K.square((y_pred - y_true))))
        return rmse
