import keras.backend  as K
import tensorflow as tf
import numpy as np

# A lot of insipration from https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py

def TripletLossNaive(N = 3, dist='sqeuclidean', margin='maxplus'):
    """
    WARNING: requires model to output shape (batch_size, N * 3)


    Arguments:
    ```
    N  --  The number of features per sample (aka 1D output size of shared model)
    dist   -- the type of space to model; `sqeuclidean` or `euclidean`
    margin -- the margin type to take; `maxplus` or `softplus`
    ```

    Returns:
    ```
    loss -- def, returns actual loss
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- concatenated tensor with N * 3 items in axis -1, each N being:
            [0:N]   anchor -- the encodings for the anchor data
            [N:2N]  positive -- the encodings for the positive data (similar to anchor)
            [2N:3N] negative -- the encodings for the negative data (different from anchor)
    ```
    """
    def stub(y_true, y_pred):
        anchor = y_pred[:,0:N]
        positive = y_pred[:,N:N*2]
        negative = y_pred[:,N*2:N*3]
        positive_distance = K.square(anchor - positive)
        negative_distance = K.square(anchor - negative)
        if dist == 'euclidean':
            positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
            negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
        elif dist == 'sqeuclidean':
            positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
            negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
        loss = positive_distance - negative_distance
        if margin == 'maxplus':
            loss = K.maximum(0.0, 1 + loss)
        elif margin == 'softplus':
            loss = K.log(1 + K.exp(loss))
        return K.mean(loss)
    return stub


def BPRTripletLoss(N = 3):
    """  
    NOTE: In testing this seems to perform worse than Naive euclidean triplet loss
    WARNING: requires model to output shape (batch_size, N * 3)

    Arguments:
    ```
    N  --  The number of dimension   
    ```

    Returns:
    ```
    loss -- def, returns actual loss
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- concatenated tensor with N * 3 items in axis -1, each N being:
            [0:N]   anchor -- the encodings for the anchor data
            [N:2N]  positive -- the encodings for the positive data (similar to anchor)
            [2N:3N] negative -- the encodings for the negative data (different from anchor)
    ```
    """
    def stub(y_true, y_pred):
        anchor = y_pred[:,0:N]
        positive = y_pred[:,N:N*2]
        negative = y_pred[:,N*2:N*3]
        
        loss = 1.0 - K.sigmoid(
            K.sum(anchor * positive, axis=-1, keepdims=True) -
            K.sum(anchor * negative, axis=-1, keepdims=True))
        
        return K.mean(loss)
    return stub

def LosslessTripletLoss(N = 3, beta = 3, epsilon=1e-8):
    """  
    NOTE: In testing this seems to perform worse than Naive euclidean triplet loss
    WARNING: requires model to output shape (batch_size, N * 3)

    Arguments:
    ```
    N  --  The number of dimension 
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)    
    ```

    Returns:
    ```
    loss -- def, returns actual loss
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- concatenated tensor with N * 3 items in axis -1, each N being:
            [0:N]   anchor -- the encodings for the anchor data
            [N:2N]  positive -- the encodings for the positive data (similar to anchor)
            [2N:3N] negative -- the encodings for the negative data (different from anchor)
    ```
    """
    def stub(y_true, y_pred):
        anchor = y_pred[:,0:N]
        positive = y_pred[:,N:N*2]
        negative = y_pred[:,N*2:N*3]
        
        positive_distance = K.sum(K.square(anchor - positive), axis=1)
        negative_distance = K.sum(K.square(anchor - negative), axis=1)
        
        #Non Linear Values  
        
        # -ln(-x/N+1)
        pos_dist = -K.log(-(positive_distance/beta)+1+epsilon)
        neg_dist = -K.log(-((N-negative_distance)/beta)+1+epsilon)
        
        # compute loss
        loss = neg_dist + pos_dist
        
        return loss
    return stub