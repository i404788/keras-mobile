from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras.models import Model
from keras import backend as K

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import to_categorical
from keras.utils import plot_model


def get_model():
    input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3))(input_img)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(input_img, x)
    return model

model = get_model()
model.summary()
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file='baseline-classify.png', show_shapes=True)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Train the model
model.fit(X_train / 255.0, to_categorical(Y_train),
          batch_size=64,
          shuffle=True,
          epochs=50,
          validation_data=(X_test / 255.0, to_categorical(Y_test)),
          callbacks=[EarlyStopping(min_delta=0.001, patience=5),
            TensorBoard(log_dir='logs/cifar-keras-cae', histogram_freq=5)])

# Evaluate the model
scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

print('Baseline CIFAR-10: Accuracy: %.3f, Params: %d' % (scores[1], model.count_params()))