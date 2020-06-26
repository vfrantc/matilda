import numpy as np
import tensorflow as tf
from matilda.transform import make_filter_bank


class Harmonic():
    pass


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
        groups = tf.split(x_input, axis=3, num_or_size_splits=x_input.shape[-1])

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


class HarmonicCombine(tf.keras.layers.Layer):

    def __init__(self, filters, activation=None, **kwargs):
        super().__init__(**kwargs)
        self._filters = filters
        if activation is not None:
            self._activation = tf.keras.activations.get(activation)
        else:
            self._activation = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(1, 1, input_shape[-1], self._filters),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.built = True

    def call(self, x_input, training=False):
        x = tf.nn.conv2d(x_input, filters=self.kernel, strides=(1, 1, 1, 1), padding='SAME')
        if self._activation is not None:
            x = self._activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self._filters,
                       'activation': self._activation})
        return config

'''
    def gen_harmonic_params(ni, no, k, normalize=False, level=None, linear=False):
        nf = k**2 if level is None else level * (level+1) // 2
        paramdict = {'conv': utils.dct_params(ni, no, nf) if linear else utils.conv_params(ni*nf, no, 1)}
        if normalize and not linear:
            paramdict.update({'bn': utils.bnparams(ni*nf, affine=False)})
        return paramdict
'''

'''
def lin_harmonic_block(x, params, base, mode, stride=1, padding=1):
    filt = torch.sum(params[base + '.conv'] * params['dct'][:x.size(1), ...], dim=2)
    y = F.conv2d(x, filt, stride=stride, padding=padding)
    return y
'''


class LinearHarmonic(tf.keras.layers.Conv2D):
    # TODO: Adopt the implementation from pytorch implementation above

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.filter_bank = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        self.filter_bank = tf.Variable(initial_value=make_filter_bank(ftype=ftype, n=n, level=level),
                                       trainable=False)

        self.built = True

    def call(self, x_input, training=False):
        filt = torch.sum(params[base + '.conv'] * params['dct'][:x.size(1), ...], dim=2)
        y = F.conv2d(x, filt, stride=stride, padding=padding)

        # split input
        groups = tf.split(x_input, axis=3, num_or_size_splits=x_input.shape[-1])

        # convolve every input channel with the filter bank
        conv_groups = [tf.nn.conv2d(input=group,
                                    filters=self.filter_bank,
                                    strides=self._strides,
                                    padding='SAME') for group in groups]
        # concatenate output feature maps
        return tf.concat(conv_groups, axis=3)
