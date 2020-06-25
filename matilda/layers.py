import numpy as np
import tensorflow as tf
from .transform import make_filter_bank


class HarmonicTransform(tf.keras.layers.Layer):
    '''Performs orthogonal transform'''
    def __init__(self, ftype='dct', n=3, level=None, strides=(1, 1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self._ftype = ftype
        self._n = n
        self._level = level
        self._strides = strides
        self.filter_bank = tf.Variable(initial_value=make_filter_bank(ftype=ftype, n=n, level=level),
                                       trainable=False)

    def build(self, input_shape):
        self.built = True

    def call(self, x_input, training=False):
        # split input
        groups = tf.split(x_input, axis=3, num_or_size_splits=3)

        # convolve every input channel with the filter bank
        conv_groups = [tf.nn.conv2d(input=group,
                                    filters=self.filter_bank,
                                    strides=self._strides,
                                    padding='SAME') for group in groups]
        # concatenate output feature maps
        return tf.concat(conv_groups, axis=3)

    def get_config(self):
        config = super().get_config()
        config.update({'ftype': self._ftype,
                       'n': self._n,
                       'level': self._level,
                       'strides': self._strides})

class HarmonicCombine():
    pass

class LinHarmonic():
    pass

class DeformLayer():
    pass

class AlphaLayer():
    pass

