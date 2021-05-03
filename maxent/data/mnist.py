"""Utils for using the MNIST dataset."""

import numpy as np
import tensorflow as tf

__all__ = (
    'load_mnist',
    'evaluate',
)


def pooling(x, size):
    # x shape: [None, width, height]
    x = tf.expand_dims(x, axis=-1)
    x = tf.image.resize(x, size)
    return x  # shape: [None, size[0], size[1], 1]


def process_data(X, y, image_size, binarize, minval, maxval):
    X = pooling(X, image_size)
    X = X / 255.
    if binarize:
        X = tf.where(X < 0.5, minval, maxval)
    else:
        X = X * (maxval - minval) + minval
    X = tf.reshape(X, [-1, image_size[0] * image_size[1]])
    y = tf.one_hot(y, 10)
    return tf.cast(X, tf.float32), tf.cast(y, tf.float32)


def evaluate_classifier(model, X, y):
    yhat = model.predict(X)
    acc = np.mean(np.argmax(y, axis=-1) == np.argmax(yhat, axis=-1))
    return acc


def evaluate_autoencoder(model, X):
    X_recon = model.predict(X)
    acc = np.mean(X == X_recon)
    return acc


def evaluate(model, X, y=None):
    if isinstance(X, tf.Tensor):
        X = X.numpy()

    if y is None:
        return evaluate_autoencoder(model, X)

    if isinstance(y, tf.Tensor):
        y = y.numpy()
    return evaluate_classifier(model, X, y)


def load_mnist(image_size, binarize, minval, maxval):
    """
    Parameters
    ----------
    image_size : (int, int)
    binarize : bool
    minval : float
    maxval : float

    Returns
    -------
    (x_train, y_train), (x_test, y_test), all are float32 np.ndarray.
    x has shape [num_data] + image_size, and y has shape [num_data, 10].
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = process_data(x_train, y_train, image_size, binarize,
                                    minval, maxval)
    x_test, y_test = process_data(x_test, y_test, image_size, binarize,
                                  minval, maxval)
    return (x_train, y_train), (x_test, y_test)
