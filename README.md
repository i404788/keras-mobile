# Keras Mobile
Fast &amp; Compact keras blocks and layers for use in mobile applications

### Currently Implemented:
* `ShuffleBasic` and `ShuffleStride` from ShuffleNet-V2
* `SeperableConvBlock` from MnasNet
* `MobileConvBlock` used in MnasNet & MobileNetV2
    * Generalization of Conv blocks used in both networks
* `GroupConv` group convolutional layer (implemented as a block)
* `MatrixSwish|ScalarSwish` Swish with learnable `beta` parameter
* `Swish|Mish` *ish activations as Layers
* `log_softmax` for KL Divergence distilation
* `cRelu` concatenated ReLU, which is `relu(x) || relu(-x)`
* `InstanceLayerNormalization` 
* `AdaptiveInstanceLayerNormalization` from UGATIT
* `ResnetBlock`
* `ApesBlock` from ApesNet
* `VectorQuantizer` from VQ-VAE
* `TripletLossNaive`, `LosslessTripletLoss` and `BPRTripletLoss` for sparse embedding/single-shot classification
* `Radam|LAMBOptimizer` for stable, fast training.

### Current Namespaces:
* `keras_mobile.blocks.conv.{SeperableConvBlock|MobileConvBlock|GroupConv|ShuffleBasic|ShuffleStride|ResnetBlock|ApesBlock}`
* `keras_mobile.layers.activations.{MatrixSwish|ScalarSwish|cRelu}`
* `keras_mobile.layers.normalization.{InstanceLayerNormalization|AdaptiveInstanceLayerNormalization}`
* `keras_mobile.layers.conv.{VectorQuantizer}`
* `keras_mobile.functions.activations.{Swish|Mish|log_softmax}`
* `keras_mobile.functions.mutations.{channel_split|channel_shuffle}`
* `keras_mobile.functions.loss.{TripletLossNaive|LosslessTripletLoss|BPRTripletLoss}`
* `keras_mobile.functions.optimizers.{Radam|LAMBOptimizer}`

### Pre-made models:
All models are in `examples/`, on execution a model graph will be generated as PNG.
The accuracy in our benchmark is the best validation accuracy with the script run as is.
We allow 50 epochs (batch size of 32), with early stopping (p=5) and default Adam optimizer.
```
Classify
Model               Dataset     Accuracy    Params

Baseline            CIFAR-10    0.797       776394

Shuffle(b=.9)       CIFAR-10    0.721       13270
Shuffle(b=1.5)      CIFAR-10    0.789       27576
Shuffle(b=2.)       CIFAR-10    0.804       42754
Shuffle(b=10,d=.2)  CIFAR-10    0.829       743650

Mobile (b=1.5)      CIFAR-10    0.664       13030
```

Clustering is done with 150x150 samples, 10 epochs, batch size 64, Radam optimizer.
Currently accuracy is not yet measured, and this table is just to signify which configurations work 'well enough'.
```
Clustering (triplet loss)
Model           Dataset       Loss            Params
Shuffle(b=2.0)  MNIST         NaiveLoss()     ±44000 
```

Feel free to improve these models, and create an PR. Changes are welcome as long is it's not a great cost on complexity or params count.

### Notes:
#### Inference
You can use [keras-to-tensorflow](https://github.com/amir-abdi/keras_to_tensorflow) and `model.save("model.h5")` to accelerate and compress the model for inference.

#### Tensorflow Backend
In some modules tensorflow is used to fill in gaps from the keras abstracted backend.
So it is preferred to use the tensorflow backend, PRs to resolve these limitations are welcome.

#### Block interface
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


### Attribution & Contribution
Some modules have been taken/modified from other repositories; these will have a `# by <name>, (<license>, <year>)` above them.

Everything without attribution should be mostly written by contributors of this repository.

If you create an PR adding any feature make sure to add the same annotation (if you want attribution).

As the end-user you are free to take any component from this repository, but we request you keep the attribution.