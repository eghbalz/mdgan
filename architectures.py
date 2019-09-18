import lasagne
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import rectify, LeakyRectify
import theano

class ScaledSig(object):
    def __init__(self, scale=5., shift=2.5):
        self.scale = scale
        self.shift = shift
    def __call__(self, x):
        return self.scale * theano.tensor.nnet.sigmoid(x) - self.shift
class ScaledTanh(object):
    def __init__(self, scale=6.):
        self.scale = scale
    def __call__(self, x):
        return self.scale * lasagne.nonlinearities.tanh(x)

def build_generator(noise_var=None,zdim=None):
    layer = InputLayer(shape=(None, zdim), input_var=noise_var,init=lasagne.init.Normal(std=0.02, mean=0))
    layer = DenseLayer(layer, num_units=128, nonlinearity=rectify, W=lasagne.init.Normal(std=0.02, mean=0))
    layer = DenseLayer(layer, num_units=128, nonlinearity=rectify, W=lasagne.init.Normal(std=0.02, mean=0))
    layer = DenseLayer(layer, num_units=2, nonlinearity=ScaledTanh(), W=lasagne.init.Normal(std=0.02, mean=0))
    return layer

def build_discriminator(input_var=None,bot_dim=None):
    #
    layer = InputLayer(shape=(None, 2), input_var=input_var)
    layer = DenseLayer(layer, num_units=128, nonlinearity=LeakyRectify(0.2), W=lasagne.init.Normal(std=0.02, mean=0))
    layer = DenseLayer(layer, num_units=128, nonlinearity=LeakyRectify(0.2), W=lasagne.init.Normal(std=0.02, mean=0))
    layer = DenseLayer(layer, num_units=bot_dim, nonlinearity=ScaledSig(), W=lasagne.init.Normal(std=0.02, mean=0))
    return layer

def build_lambda_network(lambda_input_var, bot_dim):
    layer = lasagne.layers.InputLayer(shape=(None,), input_var=lambda_input_var)
    layer = lasagne.layers.ReshapeLayer(layer, (-1, 1))
    layer = lasagne.layers.DenseLayer(layer, num_units=bot_dim, nonlinearity= ScaledSig(), W=lasagne.init.Normal(std=0.02, mean=0))
    return layer
