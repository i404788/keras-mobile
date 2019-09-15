from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras_mobile.blocks.conv import SeperableConvBlock, MobileConvBlock, ApesBlock, ResnetBlock, ShuffleStride, ShuffleBasic
from keras_mobile.layers.normalization import InstanceLayerNormalization
from keras_mobile.callbacks.tensorboard import TensorBoardImageComparison
from keras.layers import Input, UpSampling2D, ReLU, Conv2D
from keras import Model

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

from keras.utils import plot_model

def get_model():
    # Swishf = Swish(1.0)
    
    inp = x = Input(shape=(32,32,3))
    x = Conv2D(32, (3,3), padding='same')(x)
    for i in range(3):
        x = ShuffleBasic(32//(2 ** i), 0.9)(x)
        x = ShuffleStride(32//(2 ** i), 0.9)(x)
        x = ReLU()(x)
    
    x = Conv2D(8, (3,3), padding='same')(x)

    encoded = x
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inp, decoded)
    return autoencoder, inp, encoded, decoded

model = get_model()[0]
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
plot_model(model, to_file='shuffle-ae.png', show_shapes=True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
model.fit(x_train, x_train, epochs=50, batch_size=32, validation_data=(x_test, x_test), 
          callbacks=[TensorBoard(log_dir='logs/autoencoder', histogram_freq=5),
                     TensorBoardImageComparison('logs/autoencoder-alt', 'Samples', x_test)])