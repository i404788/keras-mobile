from keras.layers import Layer
from keras.initializers import Ones
from keras.constraints import MinMaxNorm
import keras.backend as K
import tensorflow as tf


class ScalarSwish(Layer):
    # creating a layer class in keras
    def __init__(self, **kwargs):
        super(ScalarSwish, self).__init__(**kwargs)
    
    def build(self, input_shape): 
        # initialize weight matrix for each capsule in lower layer
        self.beta = self.add_weight(shape = [1], initializer = Ones(), name = 'beta', constraint=MinMaxNorm(-0.2, 2.0, 0.8))
        self.built = True
    
    def call(self, inputs):
        inputs = K.sigmoid(inputs * self.beta) * inputs
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

class MatrixSwish(Layer):
    # creating a layer class in keras
    def __init__(self, **kwargs):
        super(MatrixSwish, self).__init__(**kwargs)
    
    def build(self, input_shape): 
        # initialize weight matrix for each capsule in lower layer
        self.beta = self.add_weight(shape=list(input_shape)[1:], name = 'beta', initializer=Ones(), constraint=MinMaxNorm(-0.1, 2.0, 0.8), trainable=True)
        self.built = True
    
    def call(self, inputs):
        inputs = K.sigmoid(inputs * self.beta) * inputs
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

class cRelu(Layer):
    def __init__(self, **kwargs):
        super(cRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        super(cRelu, self).build(input_shape)

    def call(self, x):
        return tf.nn.crelu(x)

    def compute_output_shape(self, input_shape):
        """
        All axis of output_shape, except the last one,
        coincide with the input shape.
        The last one is twice the size of the corresponding input 
        as it's the axis along which the two relu get concatenated.
        """
        return (*input_shape[:-1], input_shape[-1]*2)
 