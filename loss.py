from keras import backend as K
import tensorflow as tf
import math

N_LANDMARK = 106


def normalized_mean_error(y_true, y_pred):
    '''
    normalised mean error
    '''
    y_pred = K.reshape(y_pred, (-1, N_LANDMARK, 2))
    y_true = K.reshape(y_true, (-1, N_LANDMARK, 2))
    # Distance between pupils
    interocular_distance = K.sqrt(
        K.sum((y_true[:, 38, :] - y_true[:, 92, :]) ** 2, axis=-1))
    return K.mean(K.sum(K.sqrt(K.sum((y_pred - y_true) ** 2, axis=-1)), axis=-1)) / \
        K.mean((interocular_distance * N_LANDMARK))


# def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
#     """
#     Reference: wing loss for robust facial landmark localisation
#     with convolutional neural networks
#     """
#     x = y_true - y_pred
#     c = w * (1.0 - math.log(1.0 + w/epsilon))
#     absolute_x = K.abs(x)
#     losses = tf.where(
#         K.greater(w, absolute_x),
#         w * K.log(1.0 + absolute_x/epsilon),
#         absolute_x - c
#     )
#     loss = K.mean(K.sum(losses, axis=-1), axis=0)

#     return loss

def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    y_true = tf.reshape(y_true, [-1, N_LANDMARK, 2])
    y_pred = tf.reshape(y_pred, [-1, N_LANDMARK, 2])

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = tf.abs(x)
    losses = tf.where(
        tf.greater(w, absolute_x),
        w * tf.log(1.0 + absolute_x/epsilon),
        absolute_x - c
    )
    loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)

    return loss


def smoothL1(y_true, y_pred):
    """
    More robust to noise
    """
    THRESHOLD = K.variable(1.0)
    mae = K.abs(y_true - y_pred)
    flag = K.greater(mae, THRESHOLD)
    loss = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)), axis=-1)

    return loss



