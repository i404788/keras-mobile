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
The accuracy in our benchmark is the best validation accuracy with the script run as is.
We allow 50 epochs (batch size of 32), with early stopping (p=5) and default Adam optimizer.
```
Classify
Model           Dataset     Accuracy    Params

Baseline        CIFAR-10    0.797       776394

Shuffle(b=0.9)  CIFAR-10    0.721       13270
Shuffle(b=1.5)  CIFAR-10    0.789       27576
Shuffle(b=2.0)  CIFAR-10    0.804       42754

Mobile (b=1.5)  CIFAR-10    0.664       13030
```
Feel free to improve these models, and create an PR. Changes are welcome as long is it's not a great cost on complexity or params count.

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