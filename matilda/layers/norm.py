import tensorflow as tf

class CoeffNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 gamma_init='one',
                 beta_init='zero',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 epsilon=1e-6,
                 group=16,
                 **kwargs):
        super().__init__(**kwargs)

        self.gamma_init = tf.keras.initializers.get(gamma_init)
        self.beta_init = tf.keras.initializers.get(beta_init)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.group = group

    def build(self, input_shape):
        shape = [1 for _ in input_shape]
        shape.append(self.group)

        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs):
        input_shape = inputs.get_shape()

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1

        x = tf.reshape(inputs, (batch_size, h, w, c // self.group, self.group))
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        x = (x - mean) / std
        x = self.gamma * x + self.beta
        x = tf.reshape(x, (batch_size, h, w, c))
        return x

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'gamma_init': tf.keras.initializers.serialize(self.gamma_init),
                  'beta_init': tf.keras.initializers.serialize(self.beta_init),
                  'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
                  'group': self.group}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))