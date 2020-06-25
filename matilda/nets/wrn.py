import tensorflow as tf

weight_decay = 0.0005

def block(x, width, stride, dropout):
    o1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5,  gamma_initializer='uniform')(x)
    o1 = tf.keras.layers.Activation('relu')(o1)
    y = tf.keras.layers.Conv2D(width,
                       kernel_size=(3, 3),
                       strides=(stride, stride),
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                       use_bias=False)(o1)
    o2 = tf.keras.layers.BatchNormalization(axis=-1,
                            momentum=0.1,
                            epsilon=1e-5,
                            gamma_initializer='uniform')(y)
    if dropout > 0:
        o2 = tf.keras.layers.Dropout(dropout)(o2)
    o2 = tf.keras.layers.Activation('relu')(o2)
    z = tf.keras.layers.Conv2D(width,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      use_bias=False)(o2)
    if z.shape[-1] != x.shape[-1]:
        side_conv = tf.keras.layers.Conv2D(width,
                                  kernel_size=(3, 3),
                                  strides=(stride, stride),
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  use_bias=False)(o1)
        x = tf.keras.layers.Add()([z, side_conv])
    else:
        x = tf.keras.layers.Add()([z, x])

    return x

def group(x, n, width, stride, dropout):
    for i in range(n):
        x = block(x, width, stride if i==0 else 1, dropout)
    return x

