import numpy as np
import tensorflow as tf

class AlphaRooting(tf.keras.layers.Layer):
  def __init__(self, initial_alpha, output=16, size=3):
    super(AlphaHarmonicLayer, self).__init__()
    self._initial_alpha = initial_alpha
    self._size = size
    self._output = output
    self._filters = tf.Variable(initial_value = chebychev_filters(n=size, groups=1, expand_dim=2),
                                trainable=False)
    #self.batch_norm = tf.keras.layers.BatchNormalization()

  def build(self, input_shape):
    self.alpha = self.add_weight(shape=(1,
                                        1,
                                        1,
                                        input_shape[-1]*self._size*self._size),
                                 initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1.0),
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
    powered = tf.math.multiply(tf.sign(filtered), tf.pow(tf.abs(filtered), tf.abs(self.alpha)))
    return powered

  def get_config(self):
      config = super(AlphaHarmonicLayer, self).get_config()
      config.update({'output': self._output,
                     'size': self._size})
      return config