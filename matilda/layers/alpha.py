import numpy as np
import tensorflow as tf
from matilda.transform import make_filter_bank

class Clip(tf.keras.constraints.Constraint):

  def __init__(self, min_value, max_value=None):
    self.min_value = min_value
    self.max_value = max_value
    if self.max_value is None:
      self.max_value = -self.min_value
    if self.min_value > self.max_value:
      self.min_value, self.max_value = self.max_value, self.min_value

  def __call__(self, p):
    return tf.clip_by_value(p, self.min_value, self.max_value)

  def get_config(self):
    return {"min_value": self.min_value,
            "max_value": self.max_value}



class AlphaRooting(tf.keras.layers.Layer):

    def __init__(self, alpha=0.9, trainable=True, min_value=0.8, max_value=1.2, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._min_value = min_value
        self._max_value = max_value
        self._trainable = trainable
        self.alpha_constraint = Clip(min_value=self._min_value, max_value=self._max_value)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(1,
                                            1,
                                            1,
                                            input_shape[-1]),
                                     initializer=tf.keras.initializers.Constant(value=self._alpha),
                                     trainable=self._trainable,
                                     constraint=self.alpha_constraint)
        self.built = True

    def call(self, x_input, train=False):
        return tf.math.multiply(tf.sign(x_input), tf.pow(tf.abs(x_input), self.alpha))

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self._alpha,
                       'trainable': self._trainable,
                       'min_value': self._min_value,
                       'max_value': self._max_value})