import numpy as np
import tensorflow as tf
from matilda.transform import make_filter_bank


class ChebychevTransform(tf.keras.layers.Layer):
    '''Performs orthogonal transform'''

    def __init__(self, n=4, level=None, strides=(1, 1, 1, 1), padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self._n = n
        self._level = level
        self._strides = strides
        self._padding = padding
        if level is None:
            self._indexes = None
        else:
            self._indexes = [i*n+j for i, j in level]

    def build(self, input_shape):
        self.built = True

    def call(self, x_input, training=False):
        batch_size, im_height, im_width, im_channels = x_input.shape.as_list()
        if batch_size is None:
            batch_size = -1
        features = self.extract_cheb_features(x_input,
                                              N=self._n,
                                              M=self._n,
                                              strides=self._strides,
                                              padding=self._padding.upper())
        if self._indexes is not None:
            out_dim = im_channels * len(self._indexes)
            features = tf.reshape(features, batch_size, im_height, im_width, im_channels, self._n*self._n)
            features = tf.gather(features, indices=self._indexes, axis=-1)
        else:
            out_dim = im_channels * self._n * self._n

        return tf.reshape(features, (batch_size, im_height, im_width, out_dim))

    @tf.function
    def extract_cheb_features(self, image, N, M, strides=(1, 1, 1, 1), padding='SAME'):
        '''Based on: https://www.mathworks.com/matlabcentral/fileexchange/4416-2d-chebyshev-transform'''
        # Takes as an input image: batch_size x height x width x channels
        # Forms blocks of NxM and performs chebychev transform on them
        batch_size, im_height, im_width, im_channels = image.shape.as_list()
        if batch_size is None:
            batch_size = -1

        # Extract patches of size NxM
        patches = tf.image.extract_patches(image, sizes=(1, N, M, 1), strides=strides, rates=(1, 1, 1, 1),
                                           padding='SAME')
        patches = tf.reshape(patches, [-1, im_height, im_width, N, M, im_channels])

        # form the tensor of patches to transform (last two dimensions are dimensions of patches)
        patches = tf.transpose(patches, [0, 1, 2, 5, 4, 3])
        patches = tf.cast(patches, tf.complex64)

        # Chebychev transform using inverse fourier transform
        F = tf.signal.ifft(tf.concat([patches, patches[..., (N - 2):0:-1]], axis=-1))
        B = tf.concat([tf.expand_dims(F[..., 0], axis=5), 2 * F[..., 1:N]], axis=5)
        G = tf.transpose(B, [0, 1, 2, 3, 5, 4])
        F = tf.signal.ifft(tf.concat([G, G[..., (M - 2):0:-1]], axis=-1))

        return tf.math.real(tf.transpose(tf.concat([tf.expand_dims(F[..., 0], axis=5), 2 * F[..., 1:M]], axis=5), [0, 1, 2, 3, 5, 4]))

    def get_config(self):
        config = super().get_config()
        config.update({'n': self._n,
                       'level': self._level,
                       'strides': self._strides,
                       'padding': self._padding})
        return config



class SummationHarmonicLayer(tf.keras.layers.Layer):
  def __init__(self, ftype='dct', size=4):
    super(SummationHarmonicLayer, self).__init__()
    self._size = size
    self._filters = tf.Variable(initial_value = make_filter_bank(ftype=ftype, n=size),
                                trainable=False)

  def build(self, input_shape):
    # nothing to do
    pass

  def call(self, x_input, training=False):
    # split input
    groups = tf.split(x_input, axis=3, num_or_size_splits=x_input.shape[-1])
    # convolve every input channel with filter bank
    conv_groups = [tf.nn.conv2d(input = group,
                                filters=self._filters,
                                strides=(1,),
                                padding='SAME') for group in groups]

    # summarize output feature maps
    sums = tf.add_n(conv_groups)
    return sums

class HarmonicTransform(tf.keras.layers.Layer):
    '''Performs orthogonal transform'''

    def __init__(self, ftype='dct', n=3, level=None, normalize=True, strides=(1, 1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self._ftype = ftype
        self._n = n
        self._level = level
        self._strides = strides

        self.filter_bank = tf.Variable(initial_value=make_filter_bank(ftype=ftype, n=n, level=level, l1_norm=normalize),
                                       trainable=False)

    def build(self, input_shape):
        self.built = True

    def call(self, x_input, training=False):
        # split input
        print(x_input.shape)
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


class LinearHarmonic(tf.keras.layers.Conv2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 ftype='dct',
                 level=None,
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
        self.ftype = ftype
        self.level = level
        self.filter_bank = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        out_channels = self.filters
        filters = make_filter_bank(ftype=self.ftype, n=self.kernel_size[0], level=self.level)
        num_filters = filters.shape[-1]

        filters = filters[:, :, :, np.newaxis, :]
        filters = np.tile(filters, [1, 1, in_channels, out_channels, 1])
        filters = tf.convert_to_tensor(filters, dtype=tf.float32)

        self.filter_bank = tf.Variable(initial_value=filters,
                                       name='filter_bank',
                                       trainable=False)


        kernel_shape = (1, 1, in_channels, out_channels, num_filters)

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

        self.built = True

    def call(self, x_input, training=False):
        filt = tf.reduce_sum(self.kernel * self.filter_bank, axis=-1)
        conv = tf.nn.conv2d(x_input, filters=filt, strides=self.strides, padding=self.padding.upper())
        if self.use_bias:
            conv += self.bias
        return self.activation(conv)

class Harmonic(tf.keras.layers.Conv2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 ftype='dct',
                 level=None,
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
        self.ftype = ftype
        self.level = level
        self.filter_bank = None
        self.kernel = None
        self.bias = None

        filters = make_filter_bank(ftype=ftype, n=self.kernel_size[0], level=level)
        self.filter_bank = tf.Variable(initial_value=filters,
                                       trainable=False,
                                       name='filterbank')

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        out_channels = self.filters
        num_filters = self.filter_bank.shape[-1]
        kernel_shape = (1, 1, in_channels*num_filters, out_channels)

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

        self.built = True

    def call(self, x_input, training=False):
        # split input
        groups = tf.split(x_input, axis=3, num_or_size_splits=x_input.shape[-1])

        # convolve every input channel with the filter bank
        conv_groups = [tf.nn.conv2d(input=group,
                                    filters=self.filter_bank,
                                    strides=self.strides,
                                    padding=self.padding.upper()) for group in groups]
        # concatenate output feature maps
        filtered = tf.concat(conv_groups, axis=3)

        conv = tf.nn.conv2d(filtered,
                            filters=self.kernel,
                            strides=[1, 1, 1, 1],
                            padding=self.padding.upper())
        if self.use_bias:
            conv += self.bias
        return self.activation(conv)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype(np.float32) / 127.5) - 1
    x_test = (x_test.astype(np.float32) / 127.5) - 1
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        HarmonicTransform(ftype='dct', n=3, strides=(1, 1, 1, 1)),
        HarmonicCombine(32, activation='relu'),
        Harmonic(32, [3, 3], activation='relu'),
        tf.keras.layers.MaxPool2D(2, [2, 2]),
        LinearHarmonic(32, [3, 3], activation='relu'),
        LinearHarmonic(32, [3, 3], activation='relu'),
        tf.keras.layers.MaxPool2D(2, [2, 2]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))