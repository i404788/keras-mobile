from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten,ReLU, concatenate, add, BatchNormalization, GaussianNoise, SpatialDropout2D
from keras.models import Model
from keras_mobile.blocks.conv import SeperableConvBlock, MobileConvBlock, ApesBlock, ResnetBlock, ShuffleStride, ShuffleBasic
from keras_mobile.layers.normalization import InstanceLayerNormalization
from keras_mobile.layers.activations import ScalarSwish

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import to_categorical
from keras.utils import plot_model

def get_model(hp_bottleneck, out_size=10):
    inp = x = Input(shape=(32,32,3))
    x = GaussianNoise(0.1)(x)
    x = Conv2D(32, (3,3), padding='same')(x)
    for i in range(6):
        l = x
        x = ShuffleBasic(32//(2 ** i), hp_bottleneck)(x)
        x = Dropout(0.3)(x)
        x = add([x, l])
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = ShuffleStride(32//(2 ** i), hp_bottleneck)(x)
        x = ReLU()(x)

    x = ShuffleBasic(16, hp_bottleneck)(x)
    x = Flatten()(x)
    x = Dense(out_size, activation='softmax')(x)
    model = Model(inp, x)
    return model


if __name__ == "__main__":
    hp_bottleneck_size = 7.0
    model = get_model(hp_bottleneck_size)
    model.summary()
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    plot_model(model, to_file='shuffle-classify.png', show_shapes=True)


    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=64,
              shuffle=True,
              epochs=50,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=5),
                TensorBoard(log_dir='logs/shuffle-classify', histogram_freq=5)])

    # Evaluate the model
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    print('Shuffle(b=%.1f) CIFAR-10 Accuracy: %.3f, Params: %d' % (hp_bottleneck_size, scores[1], model.count_params()))