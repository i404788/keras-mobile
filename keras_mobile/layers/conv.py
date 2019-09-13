from keras.layers import Layer
import keras.backend as K

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