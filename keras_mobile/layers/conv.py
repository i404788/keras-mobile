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


class GridSlice2D(Layer):
  def __init__(self, chunk_shape=(32,32), **kwargs):
      super(GridSlice2D, self).__init__(**kwargs)
      self.chunk_shape = chunk_shape

  def build(self, input_shape):
      super(GridSlice2D, self).build(input_shape)
      self.ishape = input_shape

  def call(self, inputs):
    batch, height, width, channels = self.ishape
    # if len(x.shape.as_list()) != 4:
    #   raise ValueError('Need 2D image with channel axis and batch size (4D-tensor)')
    sx = width / self.chunk_shape[0]
    sy = height / self.chunk_shape[1]

    v = []
    for ix in range(ceil(sx)):
      for iy in range(ceil(sy)):
        size_y = min(self.chunk_shape[1], height - (self.chunk_shape[1] * iy))
        size_x = min(self.chunk_shape[0], width - (self.chunk_shape[0] * ix))
        offset_x = self.chunk_shape[0] * ix
        offset_y = self.chunk_shape[1] * iy
        placeholder = K.spatial_2d_padding(
                                          inputs[:, offset_y:size_y + offset_y, offset_x:size_x + offset_x, :],
                                          padding=((0, self.chunk_shape[1] - size_y), (0, self.chunk_shape[0]- size_x)),
                                          data_format='channels_last')
        v.append(placeholder)
    return v

  def compute_output_shape(self, input_shape):
      batch, height, width, channels = self.ishape
      sx = ceil(width / self.chunk_shape[0])
      sy = ceil(height / self.chunk_shape[1])
      # Repeat shape sx * sy times
      return (sx*sy) * [(batch, self.chunk_shape[1], self.chunk_shape[0], channels)]

class GridReduction(Layer):
    """
    DevDroplets.ga Research (c) 2019

    Reduce a variable sized image (sliced into a grid of `AxB` chunks) to a single `AxB` chunk with `filters*9` channels

    Usage:
    * Get a `CxD` image, split it into `AxB` chunks (See GridSlice2D)
    * When the image size changes, you call update_grid with the new grid size
    * To train:
    * * Get your images
    * * Sort them by size
    * * Optional: Preprocess using `functinos.mutations.slice_2d`
    * * Loop:
    * * * select images of same size (after preprocessing if used)
    * * * update_grid
    * * * train_on_batch with images of the same input shape

    """
    def __init__(self, grid_shape=(2,2), scale_max=16, filters=16, **kwargs):
        super(GridReduction, self).__init__(**kwargs)
        self.grid_shape = grid_shape
        self.scale_max = scale_max
        self.filters = filters
    
    def build(self, input_shape):
        super(GridReduction, self).build(input_shape)
        assert len(input_shape) is self.grid_shape[0] * self.grid_shape[1]

        self.c = []
        for s in range(self.scale_max+1):
            _c = Conv2D(self.filters, (3,3), padding='same')
            if s is 0:
                _c.build(input_shape[0])
            else:
                batch, height, width, channels = input_shape[0]
                _c.build([batch, height, width, self.filters * 9])
            _c.extend(self._c.trainable_weights)
            self.c.append(_c)

    # Used to externally update the image size
    def update_grid(self, grid_shape):
        self.grid_shape = grid_shape

    # reshape x to (?, n) shape
    def reshape(x, n):
        return [list(z) for z in zip(*[iter(x)]*n)]

    # 3x3 superconvolution around (x,y)
    def superconvolution(_sliced, c=(1,1)):
        x, y = c
        l = []
        for i in range(3):             
            for j in range(3):
                l.append(_sliced[x-(i-1)][y-(j-1)])
                
        assert len(l) is 9
        return K.concatenate(l)
    
    def column(matrix, i):
        return [row[i] for row in matrix]
    
    def call(self, inputs):
        sy, sx = self.grid_shape
        sliced = inputs
        while sx > 1 or sy > 1:
            scale = max(sx, sy)
            
            # convconcat
            if sx > 2 and sy > 2:
                # Apply scaled conv
                sliced = [self.c[scale-2](x) for x in sliced]
                _sliced = reshape(sliced, sx)
                
                # apply convconcat
                # can be visualized as a conv of 3x3
                _new = []
                for i, v0 in enumerate(_sliced):
                    for j, v1 in enumerate(_sliced[i]):
                        x_check = i != 0 and i != len(_sliced)-1
                        y_check = j != 0 and j != len(_sliced[i])-1
                        if x_check and y_check:
                            _new.append(superconvolution(_sliced, (j,i)))
                    
                sliced = _new
                sx = sx-2
                sy = sy-2
                continue
        
            _sliced = reshape(sliced, sx)
            if sx is 2:
                # Mean each pair in y
                sliced = [K.mean(x, axis=0) for x in _sliced]
                sx = 1
                continue
                
            if sy is 2:
                # Mean each pair in x
                sliced = [K.mean(column(_sliced, i), axis=0) for i in range(sx)]
                sy = 1
                continue 

        return sliced[0]

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape[0]
        return [batch, height, width, self.filters * 9]


def SharedScaledConv2D(Conv2D):
    """
        DevDroplets.ga Research (c) 2019

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