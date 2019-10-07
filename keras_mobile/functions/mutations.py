from keras.layers import Lambda, Layer
import keras.backend as K
from math import ceil

# by opconty (MIT License, 2018)
def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip])(x)
    c = Lambda(lambda z: z[:, :, :, ip:])(x)
    return c_hat, c

# by scheckmedia (MIT License, 2017)
def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

# Generates a grid of slices with size `chunk_shape` from of 4D tensor (image), padding is done with 0s (probably)
def slice_2d(x, chunk_shape=(32,32)):
    batch, height, width, channels = x.shape.as_list()
    # if len(x.shape.as_list()) != 4:
    #   raise ValueError('Need 2D image with channel axis and batch size (4D-tensor)')
    sx = width / chunk_shape[0]
    sy = height / chunk_shape[1]

    v = []
    for ix in range(ceil(sx)):
      for iy in range(ceil(sy)):
        size_y = min(chunk_shape[1], height - (chunk_shape[1] * iy))
        size_x = min(chunk_shape[0], width - (chunk_shape[0] * ix))
        offset_x = chunk_shape[0] * ix
        offset_y = chunk_shape[1] * iy
        placeholder = K.spatial_2d_padding(
                                          x[:, offset_y:size_y + offset_y, offset_x:size_x + offset_x, :],
                                          padding=((0, chunk_shape[1] - size_y), (0, chunk_shape[0]- size_x)),
                                          data_format='channels_last')
        v.append(placeholder)

    return v