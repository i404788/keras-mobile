from keras.layers import Layer, Embedding, Conv2D
import keras.backend as K
import numpy as np

class VectorQuantizer(Layer):
    """
       From VQ-VAE 
    """
    def __init__(self, num_codes, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_codes = num_codes

    def build(self, input_shape):
        super(VectorQuantizer, self).build(input_shape)
        dim = input_shape[-1]
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.num_codes, dim),
            initializer='uniform'
        )

    def call(self, inputs):
        """
            inputs.shape=[None, m, m, dim]
        """
        l2_inputs = K.sum(inputs**2, -1, keepdims=True)
        l2_embeddings = K.sum(self.embeddings**2, -1)
        for _ in range(K.ndim(inputs) - 1):
            l2_embeddings = K.expand_dims(l2_embeddings, 0)
        embeddings = K.transpose(self.embeddings)
        dot = K.dot(inputs, embeddings)
        distance = l2_inputs + l2_embeddings - 2 * dot
        codes = K.cast(K.argmin(distance, -1), 'int32')
        code_vecs = K.gather(self.embeddings, codes)
        return [codes, code_vecs]

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1], input_shape]


def SharedScaledConv2D(Conv2D):
    """
        DevDroplets research

        Using scale embeddings for multi-scale classification
        Note this might be possible using `bias` (expirement)

        Usage:
        Take a `size` (e.g. 32x32)
        Split Image of choice into chunks of that `size`
        For each chunk run it through SharedScaledConv2D
        If a chunk is not at the edge (strides=1) concatenate each neigbouring Chunk in axis -1 (channel axis)
        You should now have a new image of about `A-2*Size x B-2*Size`
        Repeating this process shoul reduce an image to a manageable size

        This will also ensure your image dimensions are a multiple of `size`. 
        When the image is compressed to the point of one axis being `2*size` wide, keep reducing only in the other axis.
        Once the image is then `2*Size x 2*Size`, you can attach a regular classifier at the end.

        The `size` you choose will depend on your use case, we recommend taking the mean of x and y dimensions, and dividing it by 10.
        For max_recursion you should take the maximal dimensions of your dataset, and then `max_r = ceil(max(x,y) / (size * 2))`.

        When running the SharedScaledConv2D make sure to use the scaled embedding at each 'compression' stage
        the `max_recursion` parameter defines how high this can go.
    """
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 max_recursion=10, **kwargs):
        super(SharedScaledConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.max_r = max_recursion

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError('SharedScaledConv2D takes 2 inputs')

        super(SharedScaledConv2D, self).build(input_shape[0])

        self.embedding_alpha = Embedding(self.max_r, np.prod(input_shape[0][1:]))
        self.embedding_alpha.build(input_shape[1])
        self.trainable_weights.extend(self.embedding_alpha.trainable_weights)


    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def call(self, inputs):
        scale_embedding = self.embedding_alpha(inputs[1])
        scale_embedding = K.reshape(scale_embedding, K.int_shape(inputs))

        # Combine Input and scale embedding
        x = scale_embedding + inputs
        x = super(SharedScaledConv2D, self).call(x)
        return x