import tensorflow as tf

def to_list(x):
    if type(x) not in [list, tuple]:
        return [x]
    else:
        return list(x)

class GroupNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 axis=-1,
                 gamma_init='one',
                 beta_init='zero',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 epsilon=1e-6,
                 group=32,
                 data_format=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.axis = to_list(axis)
        self.gamma_init = tf.keras.initializers.get(gamma_init)
        self.beta_init = tf.keras.initializers.get(beta_init)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.group = group

    def build(self, input_shape):
        self.input_spec = [tf.engine.InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]
        channel_axis = -1
        shape[channel_axis] = input_shape[channel_axis]

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
        if len(inputs) not in [2, 4]:
            raise ValueError('Inputs should have rank ' +
                             str(4) + ' or ' + str(2) +
                             '; Received input shape:', str(input_shape))
        if len(input_shape) == 4:
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1

            if c < self.group:
                raise ValueError('Input channels should be larger than group size' +
                                 '; Received input channels: ' + str(c) +
                                 '; Group size: ' + str(self.group))

            x = tf.reshape(inputs, (batch_size, h, w, self.group, ))

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': tf.keras.initializers.serialize(self.gamma_init),
                  'beta_init': tf.keras.initializers.serialize(self.beta_init),
                  'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
                  'group': self.group}
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def group_norm(x, gamma, beta, G, eps=1e-5):
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    return x*gamma + beta