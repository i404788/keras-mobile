from keras.layers import Input, Dense, Conv2D, UpSampling2D, Dropout, Flatten, ReLU, concatenate, GaussianNoise, add, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras_mobile.blocks.conv import SeperableConvBlock, MobileConvBlock, ApesBlock, ResnetBlock, ShuffleStride, ShuffleBasic
from keras_mobile.functions.loss import LosslessTripletLoss, TripletLossNaive, BPRTripletLoss
from keras_mobile.callbacks.tensorboard import TensorBoardModelEmbedding
from keras_mobile.functions.optimizers import LAMBOptimizer, RAdam
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import random
import os
from itertools import permutations


# N: Embedding output dim
# (x,x,x) -> ϕ -> concat-> (3*x)
def get_training_model(classifier, N=10):
    anchor_in = Input((28, 28, 1))
    anchor_out = classifier(anchor_in)

    positive_in = Input((28, 28, 1))
    positive_out = classifier(positive_in)

    negative_in = Input((28, 28, 1))
    negative_out = classifier(negative_in)

    triplet_out = concatenate([anchor_out, positive_out, negative_out])

    model = Model(inputs=[anchor_in, positive_in,
                          negative_in], outputs=triplet_out)

    lamb = RAdam(learning_rate=0.0005)
    model.compile(lamb, TripletLossNaive(N=N))

    return model


def euclid_distance(a, b):
    return K.sqrt(K.sum(K.square(a - b), axis=-1))


def generate_triplet(x, y,  ap_pairs=10, an_pairs=10):
    data_xy = tuple([x, y])

    trainsize = 1

    triplet_train_pairs = []
    y_triplet_pairs = []
    #triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        # Generating Anchor-Positive pairs
        A_P_pairs = random.sample(
            list(permutations(same_class_idx, 2)), k=ap_pairs)
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        #Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            y_Anchor = data_xy[1][ap[0]]
            Positive = data_xy[0][ap[1]]
            y_Pos = data_xy[1][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                y_Neg = data_xy[1][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
                y_triplet_pairs.append([y_Anchor, y_Pos, y_Neg])
                # test

    return np.array(triplet_train_pairs), y_triplet_pairs

# x(28,28,1) -> y(out_size)


def get_model(hp_bottleneck, out_size=10):
    inp = x = Input(shape=(28, 28, 1))
    x = GaussianNoise(0.1)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    for i in range(4):
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
    x = Dense(out_size, activation='sigmoid', name='classifier_dense')(x)
    model = Model(inp, x, name='classifier')
    return model, inp, x


def prepare_dataset():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_train /= 255
    print(X_train.shape)

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_test /= 255
    print(X_test.shape)

    x_tri, _ = generate_triplet(X_train, Y_train, ap_pairs=150, an_pairs=150)

    return [x_tri[:, 0, :], x_tri[:, 1, :], x_tri[:, 2, :]], [np.zeros((x_tri.shape[0],))], X_test, Y_test


if __name__ == "__main__":
    _model, _inp, _out = get_model(2.0)
    _model.summary()
    plot_model(_model, to_file='mnist_embedding_classifier.png',
               show_shapes=True)

    model = get_training_model(_model)
    plot_model(model, to_file='mnist_embedding.png', show_shapes=True)

    x_train, y_train, x_test, y_test = prepare_dataset()

    b_size = 32
    model.fit(x_train, y_train, batch_size=b_size, epochs=500, callbacks=[
              TensorBoardModelEmbedding('logs', 5, _model, x_test[:500], y_test[:500], True)])

    # Check tensorboard for results


# Notes:
# * There are many implementations, so it will be a lot of testing in tensoboard to see which works the best in this case
# * * Some implementations divide the mean loss by the amount of ~0.0 loss values