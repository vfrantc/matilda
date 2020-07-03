import tensorflow as tf

weight_decay = 0.0005

def block(x, width, stride, dropout):
    o1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5,  gamma_initializer='uniform')(x)
    o1 = tf.keras.layers.Activation('relu')(o1)
    y = tf.keras.layers.Conv2D(width,
                       kernel_size=(4, 4),
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
                      kernel_size=(4, 4),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      use_bias=False)(o2)
    if z.shape[-1] != x.shape[-1]:
        side_conv = tf.keras.layers.Conv2D(width,
                                  kernel_size=(4, 4),
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


def wrn4(input_shape, depth=16, width=8, num_classes=10, dropout=0.3):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Conv2D(16, kernel_size=4, activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    for width, stride in zip(widths, [1, 2, 2]):
        x = group(x, n, width, stride, dropout=dropout)

    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.1,
                                           epsilon=1e-5,
                                           gamma_initializer='uniform')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    y = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation='softmax',
                                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
