from tensorflow.keras import regularizers, initializers

class LayersUtilities:

    @staticmethod
    def regularization(name='L2'):

        if name == "L2":

            reg = regularizers.l2(0.01)

        elif name == "L1":

            reg = regularizers.l1(0.01)

        else:

            reg = None

        return reg

    @staticmethod
    def kernel_initializer(kernel_initializer_info):

        if kernel_initializer_info['name'] == 'random_normal':

            init = initializers.RandomNormal(mean=kernel_initializer_info['mean'], stddev=kernel_initializer_info['stddev'])

        elif kernel_initializer_info['name'] == 'glorot_uniform':

            init = initializers.GlorotUniform()

        else:

            init = initializers.GlorotUniform()

        return init
