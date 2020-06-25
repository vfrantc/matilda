import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K
from matilda.transform import chebychev_filters
from matilda.transform import dct_filters

def harmonic_block(output_channels=16, input_shape=(32, 32, 3), transform='dct'):
    '''
    There are no group convolutions in Keras, so it can be done like this:
        1. Take input image bitch_sizex32x32x3.
        2. Split image to its channels batch_sizex32x32x1
        3. Convolve each image with one of the 9 filters, kernel 3x3x1x9
        4. Concatenate results to get batch_sizex32x32x27
        5. Convolve the input with kernel of size 1x1x27x16
        6. Get the output of size batch_sizex32x32x16
    '''
    input = layers.Input(shape=input_shape)
    # Red channel of the image
    r = layers.Lambda(lambda x: x[:, :, :, 0])(input)
    r = layers.Reshape((input_shape[0], input_shape[1], 1))(r)

    # Green channel of the image
    g = layers.Lambda(lambda x: x[:, :, :, 1])(input)
    g = layers.Reshape((input_shape[0], input_shape[1], 1))(g)

    # Blue channel of the image
    b = layers.Lambda(lambda x: x[:, :, :, 2])(input)
    b = layers.Reshape((input_shape[0], input_shape[1], 1))(b)

    if transform == 'cheb':
        dct0 = chebychev_filters(n=3, groups=1, expand_dim=2)
    else:
        dct0 = dct_filters(n=3, groups=1, expand_dim=2)

    filtered_r = layers.Lambda(lambda x: K.conv2d(x, dct0, strides=(1,), padding='valid', dilation_rate=(1,)))(r)
    filtered_g = layers.Lambda(lambda x: K.conv2d(x, dct0, strides=(1,), padding='valid', dilation_rate=(1,)))(g)
    filtered_b = layers.Lambda(lambda x: K.conv2d(x, dct0, strides=(1,), padding='valid', dilation_rate=(1,)))(b)
    filtered = layers.Concatenate(name="conv")([filtered_r, filtered_g, filtered_b])
    conv = layers.Conv2D(filters=output_channels, kernel_size=1, activation='relu', name='dct0.conv')(filtered)
    model = models.Model(inputs=input, outputs=conv)
    return model


@tf.function
def myactivation(x):
  return tf.math.multiply(x, tf.nn.relu(x))

class MyActivation(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super(MyActivation, self).__init__(**kwargs)

  def build(self, input_shape):
    # Nothing has to be done?
    pass

  def call(self, inputs, training=False):
    return myactivation(inputs)

  def get_config(self):
    config = super(MyActivation, self).get_config()
    return config

class HarmonicLayer(tf.keras.layers.Layer):
  def __init__(self, output=16, size=3):
    super(HarmonicLayer, self).__init__()
    self._size = size
    self._output = output
    self._filters = tf.Variable(initial_value = chebychev_filters(n=size, groups=1, expand_dim=2),
                                trainable=False)

  def build(self, input_shape):
    self.w = self.add_weight(shape=(1,
                                    1,
                                    input_shape[-1]*self._size*self._size,
                                    self._output),
                             initializer='random_normal',
                             trainable=True)

  def call(self, x_input, training=False):
    # split input
    groups = tf.split(x_input, axis=3, num_or_size_splits=3)
    # convolve every input channel with filter bank
    conv_groups = [tf.nn.conv2d(input = group,
                                filters=self._filters,
                                strides=(1,),
                                padding='SAME') for group in groups]
    # concatenate output feature maps
    filtered = tf.concat(conv_groups, axis=3)
    # it think I need regular convolution, not depthwise one
    # depthwise one gives output of the size 1x96x96x432, which is not what I want
    # tf.nn.depthwise_conv2d(filtered, filter=self.w, strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.conv2d(filtered, filters=self.w, strides=(1, 1, 1, 1), padding='SAME')

  def get_config(self):
      config = super(HarmonicLayer, self).get_config()
      config.update({'output': self._output,
                     'size': self._size})
      return config

class AlphaHarmonicLayer(tf.keras.layers.Layer):
  def __init__(self, output=16, size=3):
    super(AlphaHarmonicLayer, self).__init__()
    self._size = size
    self._output = output
    self._filters = tf.Variable(initial_value = chebychev_filters(n=size, groups=1, expand_dim=2),
                                trainable=False)

  def build(self, input_shape):
    self.alpha = self.add_weight(shape=(1,
                                        1,
                                        1,
                                        input_shape[-1]*self._size*self._size),
                                 initializer='ones',
                                 trainable=True)
    self.w = self.add_weight(shape=(1,
                                    1,
                                    input_shape[-1]*self._size*self._size,
                                    self._output),
                             initializer='random_normal',
                             trainable=True)

  def call(self, x_input, training=False):
    # split input
    groups = tf.split(x_input, axis=3, num_or_size_splits=3)
    # convolve every input channel with filter bank
    conv_groups = [tf.nn.conv2d(input = group,
                                filters=self._filters,
                                strides=(1,),
                                padding='SAME') for group in groups]

    # concatenate output feature maps
    filtered = tf.concat(conv_groups, axis=3)

    # at this point filtered contains stack of filter responses, for each channel
    powered = tf.pow(filtered, self.alpha)

    return tf.nn.conv2d(powered, filters=self.w, strides=(1, 1, 1, 1), padding='SAME')

  def get_config(self):
      config = super(AlphaHarmonicLayer, self).get_config()
      config.update({'output': self._output,
                     'size': self._size})
      return config