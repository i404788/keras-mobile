# Keras Mobile
Fast &amp; Compact keras blocks and layers for use in mobile applications

### Currently Implemented:
* `ShuffleBasic` and `ShuffleStride` from ShuffleNet-V2
* `SeperableConvBlock` from MnasNet
* `MobileConvBlock` used in MnasNet & MobileNetV2
    * Generalization of Conv blocks used in both networks
* `GroupConv` group convolutional layer (implemented as a block)
* `MatrixSwish|ScalarSwish` Swish with learnable `beta` parameter
* `Swish` takes a constant scalar for `beta` 
* `InstanceLayerNormalization` 
* `AdaptiveInstanceLayerNormalization` from UGATIT
* `ResnetBlock`
* `ApesBlock` from ApesNet
* `VectorQuantizer` from VQ-VAE

### Current Namespaces:
* `keras_mobile.blocks.conv.{SeperableConvBlock|MobileConvBlock|GroupConv|ShuffleBasic|ShuffleStride|ResnetBlock|ApesBlock}`
* `keras_mobile.layers.activations.{MatrixSwish|ScalarSwish}`
* `keras_mobile.layers.normalization.{InstanceLayerNormalization|AdaptiveInstanceLayerNormalization}`
* `keras_mobile.layers.conv.{VectorQuantizer}`
* `keras_mobile.functions.activations.{Swish}`
* `keras_mobile.functions.mutations.{channel_split|channel_shuffle}`

### Pre-made models:
All models are in `examples/`, on execution a model graph will be generated as PNG
* `mobile-cae.py`
    * A convolutional autoencoder using ShuffleNet as Encoder and ApesBlocks as decoders
    * Accuracy TBD
    * ~16.6k parameters (15.5k Trainable)


### Notes:
Blocks are defined as `def(config) -> ( def(tensor) -> tensor )`, unlike layers these can be reused without sharing weights.

For example:
```py
from keras.layers import Input
from keras_mobile.blocks.conv import SeperableConvBlock

# Create some input x and some network
x = Input((32,32,3))
x = SeperableConvBlock()(x)

sep16block = SeperableConvBlock(16)
x = sep16block(x)
x = sep16block(x)
...
```