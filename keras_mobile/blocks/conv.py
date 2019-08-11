from keras.layers import Conv2D, BatchNormalization, Input, Conv2DTranspose, MaxPooling2D, UpSampling2D, DepthwiseConv2D, PReLU, LeakyReLU
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, add, ReLU
import keras.backend as K

# Emulate class behaviour for parameterization


def SeperableConvBlock(output_filters=None, ReLU_Max=None, strides=(1,1)):    
    """
    ```
    output_filters: int, size of last axis output
    ReLU_Max: float, max value as output of a ReLU in this block
    strides: int/tuple-int, same as in keras.layers.Conv2D
    ```

    From MnasNet https://arxiv.org/pdf/1807.11626.pdf
    Also used in MobileConvBlock as subblock
    """
    def stub(x):
        x = DepthwiseConv2D((3,3))(x)
        x = ReLU(max_value=ReLU_Max)(x)
        x = BatchNormalization()(x)
        
        if output_filters is None:
            x = Conv2D(K.shape(x)[-1], (1,1), strides=strides)(x)
        else:
            x = Conv2D(output_filters, (1,1), strides=strides)(x)
        x = BatchNormalization()(x)
        return x
    return stub


def MobileConvBlock(output_filters, latent_filters, ReLU_Max=None, attentionMechanism=None, strides=(1,1)):
    """
    ```
    output_filters: int, size of last axis output
    latent_filters: int, size of filters at first Conv 1x1 (see MnasNet)
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
        y = Conv2D(output_filters*6, (1,1))(x)
        y = ReLU(max_value=ReLU_Max)(y)
        y = SeperableConv2D(output_filters=output_filters, ReLU_Max=ReLU_Max, strides=strides)(y)
        if attentionMechanism is not None:
            x = attentionMechanism([x,y])
            return x
        else:
            return y
    return stub
