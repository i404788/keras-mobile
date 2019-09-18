from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten,ReLU, add, BatchNormalization
from keras.models import Model
from keras_mobile.blocks.conv import SeperableConvBlock, MobileConvBlock, ApesBlock, ResnetBlock, ShuffleStride, ShuffleBasic
from keras_mobile.layers.normalization import InstanceLayerNormalization

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import to_categorical
from keras.utils import plot_model


hp_bottleneck_size = 2.0

def get_model():
    inp = x = Input(shape=(32,32,3))
    x = Conv2D(32, (3,3), padding='same')(x)
    for i in range(4):
        filters = 32//(2 ** i)
        x = MobileConvBlock(filters, int(filters*hp_bottleneck_size), strides=(2,2))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = MobileConvBlock(filters, int(filters*hp_bottleneck_size), skipFunction=add)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inp, x)
    return model



model = get_model()
model.summary()
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file='mobile-classify.png', show_shapes=True)


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Train the model
model.fit(X_train / 255.0, to_categorical(Y_train),
          batch_size=64,
          shuffle=True,
          epochs=50,
          validation_data=(X_test / 255.0, to_categorical(Y_test)),
          callbacks=[EarlyStopping(min_delta=0.001, patience=5),
            TensorBoard(log_dir='logs/mobile-classify', histogram_freq=5)])

# Evaluate the model
scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

print('Mobile(b=%.1f) CIFAR-10 Accuracy: %.3f, Params: %d' % (hp_bottleneck_size, scores[1], model.count_params()))