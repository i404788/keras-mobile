from keras.layers import Layer
from keras.initializers import Ones
from keras.constraints import MinMaxNorm
import keras.backend as K


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
 