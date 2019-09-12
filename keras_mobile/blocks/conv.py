from keras.layers import Conv2D, BatchNormalization, Input, DepthwiseConv2D, Lambda, Concatenate
from keras.layers import GlobalAveragePooling2D, Reshape, ReLU
import keras.backend as K
from ..functions.mutations import channel_split, channel_shuffle

# Emulate class behaviour for parameterization


def SeperableConvBlock(output_filters=None, ReLU_Max=None, strides=(1,1)):    
    """
    DWConv(3x3, bn, relu) => Conv2D(1x1, bn)


    ```
    output_filters: int, size of last axis output
    ReLU_Max: float, max value as output of a ReLU in this block, if < 0, it will be Linear (no ReLU)
    strides: int/tuple-int, same as in keras.layers.Conv2D
    ```

    From MnasNet https://arxiv.org/pdf/1807.11626.pdf
    Also used in MobileConvBlock as subblock
    """
    def stub(x):
        x = DepthwiseConv2D((3,3), strides=strides)(x)
        x = BatchNormalization()(x)
        if ReLU_Max < 0:
            x = ReLU(max_value=ReLU_Max)(x)
        
        if output_filters is None:
            x = Conv2D(K.shape(x)[-1], (1,1))(x)
        else:
            x = Conv2D(output_filters, (1,1))(x)
        x = BatchNormalization()(x)
        return x
    return stub


def MobileConvBlock(output_filters, latent_filters=None, ReLU_Max=None, attentionMechanism=None, strides=(1,1)):
    r"""
    ```
       |---------------------------------------------------------------------\
     x => Conv(1x1, f: 6n, relu) -> DWConv(3x3, bn, relu) => Conv2D(1x1, bn) =?> attention
    ```

    ```
    output_filters: int, size of last axis output
    latent_filters: int, size of filters at first Conv 1x1 (see MnasNet), (default: shape[-1])
    - *_filters is generally 'k * shape[-1]' as expansion factor
    ReLU_Max: float, max value as output of a ReLU in this block
    attentionMechanism: def, a function combining 2 equi-shaped tensors (e.g. keras.layers.add)
    strides: int/tuple-int, same as in keras.layers.Conv2D
    ```

    attentionMechanism (if not None) is an keras function with the same interface as keras.layers.{add|multiply}
    if None there will be no attention added

    Stride block from MobileNetV2 (fixed )
    ```
    Strides=1: ReLU_Max=6, attentionMechanism=keras.layers.add
    Strides=2: ReLU_Max=6, strides=(2,2)
    ```

    MBConv6 from MnasNet
    ```
    latent_filters=6*output_filters, attentionMechanism=keras.layers.add
    ```

    From MobileNetV2 https://arxiv.org/pdf/1801.04381.pdf (When RELU6)
    From MnasNet https://arxiv.org/pdf/1807.11626.pdf (When RELU)
    """
    def stub(x):
        if latent_filters is None:
            latent_filters = K.int_shape(x)[-1]
        y = Conv2D(latent_filters, (1,1))(x)
        y = ReLU(max_value=ReLU_Max)(y)
        y = SeperableConvBlock(output_filters=output_filters, ReLU_Max=ReLU_Max, strides=strides)(y)
        if attentionMechanism is not None:
            x = attentionMechanism([x,y])
            return x
        else:
            return y
    return stub


def GroupConv(in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    def stub(x):
        if groups == 1:
            return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                        use_bias=False, strides=stride, name=name)(x)

        # number of intput channels per group
        ig = in_channels // groups
        group_list = []

        assert out_channels % groups == 0

        for i in range(groups):
            offset = i * ig
            group = Lambda(lambda z: z[:, :, :, offset:offset + ig], name='%s/g%d_slice' % (name, i))(x)
            group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                    use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
        return Concatenate(name='%s/concat' % name)(group_list)
    return stub

def ShuffleBasic(out_channels, bottleneck_factor):
    r"""
    ```
            /-<x>->Conv(1x1,bn,relu)->DWConv(3x3,bn)->Conv(1x1,bn,relu)-\
    <x>->ChSplit                                                      Concat(axis=-1)->ChShuffle
            |-<c_hat>---------------------------------------------------/
    ```
    """
    def stub(x):
        c_hat, c = channel_split(x)
        x = c

        bottleneck_channels = int(out_channels * bottleneck_factor)
        x = Conv2D(bottleneck_channels, (1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeperableConvBlock(bottleneck_channels, -1)
        x = ReLU()(x)
        x = Concatenate(axis=-1)([x, c_hat])
        x = Lambda(channel_shuffle)(x)
        return x
    return stub

def ShuffleStride(out_channels, bottleneck_factor, strides=(2,2)):
    r"""
    ```
        /-<y>->Conv(1x1,bn,relu)->DWConv(3x3,bn,strides)->Conv(1x1,bn,relu)-\
      <x>                                                                  Concat(axis=-1)->ChShuffle
        |-<z>->DWConv(3x3,bn,strides)-------------------->Conv(1x1,bn,rely)-/
    ```
    """
    def stub(x):
        bottleneck_channels = int(out_channels * bottleneck_factor)
        y = Conv2D(bottleneck_channels, kernel_size=(1,1), padding='same')(x)
        y = ReLU()(y)
        y = SeperableConvBlock(bottleneck_channels, ReLU_Max=-1, strides=strides)(y)
        y = ReLU()(y)

        z = SeperableConvBlock(bottleneck_channels, ReLU_Max=-1, strides=strides)(x)
        z = ReLU()(x)

        ret = Concatenate(axis=-1)([y,z])
        ret = Lambda(channel_shuffle)(ret)

        return ret
    return stub