from keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Flatten
from keras.initializers import Ones, Constant
from keras.constraints import MinMaxNorm
import keras.backend as K
import tensorflow as tf
import numpy as np


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

class AdaptiveInstanceLayerNormalization(Layer):
    # creating a layer class in keras
    def __init__(self, smoothing=True, light=False, **kwargs):
        super(AdaptiveInstanceLayerNormalization, self).__init__(**kwargs)
        self.smoothing = smoothing
        self.light = light
    
    def build(self, input_shape): 
        # initialize weight matrix for each capsule in lower layer
        self.W = self.add_weight(shape = [input_shape[-1]], initializer = Ones(), name = 'weights', constraint=MinMaxNorm())
        self.latent_size = input_shape[-1]

        # TODO: (local)Conv2D with high stride before dense? This is way to inefficient, no wonder UGATIT is 2G
        input_prod = np.prod(input_shape[1:])
        self.fc_gamma = Dense(input_shape[-1])
        self.fc_gamma.build((None, input_prod))
        self.fc_beta  = Dense(input_shape[-1])
        self.fc_beta.build((None, input_prod))
        self.flatten = Flatten()
        self.flatten.build(input_shape)
        self.trainable_weights.extend(self.fc_beta.trainable_weights)
        self.trainable_weights.extend(self.fc_gamma.trainable_weights)

        self.built = True
    
    def call(self, inputs):
        x = inputs
        # if self.light:
        #     x = GlobalAveragePooling2D()(x)
        gamma = self.flatten(x)
        gamma = self.fc_gamma(gamma)
        gamma = K.reshape(gamma, (-1,1,1,self.latent_size))
        
        beta  = self.flatten(x)
        beta  = self.fc_beta(beta)
        beta  = K.reshape(beta, (-1,1,1,self.latent_size))

        eps = 1e-5
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        x_ins = (x - ins_mean) / K.sqrt(ins_sigma + eps)

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1,2,3], keep_dims=True)
        x_ln = (x - ln_mean) / K.sqrt(ln_sigma + eps)

        rho = self.W

        if self.smoothing:
            rho = K.clip(rho - K.constant(0.1), 0.0, 1.0)
        
        x_hat = rho *  x_ins + (1 - rho) * x_ln
        x_hat = x_hat * gamma + beta

        return x_hat

    def compute_output_shape(self, input_shape):
        return input_shape
 
class InstanceLayerNormalization(Layer):
    # creating a layer class in keras
    def __init__(self, **kwargs):
        super(InstanceLayerNormalization, self).__init__(**kwargs)
    
    def build(self, input_shape): 
        # initialize weight matrix for each capsule in lower layer
        self.rho = self.add_weight(shape = [input_shape[-1]], initializer = Constant(1.0), name = 'rho', constraint=MinMaxNorm())
        self.gamma = self.add_weight(shape = [input_shape[-1]], initializer = Constant(1.0), name = 'gamma')
        self.beta = self.add_weight(shape = [input_shape[-1]], initializer = Constant(0.0), name = 'beta')
        self.built = True
    
    def call(self, inputs):
        eps = 1e-5
        ins_mean, ins_sigma = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        x_ins = (inputs - ins_mean) / K.sqrt(ins_sigma + eps)

        ln_mean, ln_sigma = tf.nn.moments(inputs, axes=[1,2,3], keep_dims=True)
        x_ln = (inputs - ln_mean) / K.sqrt(ln_sigma + eps)
        
        x_hat = self.rho *  x_ins + (1 - self.rho) * x_ln
        x_hat = x_hat * self.gamma + self.beta

        return x_hat

    def compute_output_shape(self, input_shape):
        return input_shape
 