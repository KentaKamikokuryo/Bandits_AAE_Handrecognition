import random
import numpy as np


class Utilities:

    @staticmethod
    def softmax(x):
        x[x > 709.] = 709. # max value for exp
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def softmax_temperature(x, t=1):
        x[x > 709.] = 709.  # max value for exp
        return np.exp(x / t) / np.sum(np.exp(x / t), axis=0)

    @staticmethod
    def randargmax(values):
        return np.argmax(np.random.random(values.shape) * (values == values.max()))

    @staticmethod
    def random_choice(values):
        return random.choice(list(enumerate(values)))[0]

    @staticmethod
    def cross_entropy(p, q):
        q[q == 0] = 0.00001
        return -np.sum([p[i]*np.log(q[i]) for i in range(len(p))])