import tensorflow as tf

from matilda.layers import HarmonicTransform
from matilda.layers import HarmonicCombine
from matilda.layers import LinearHarmonic

weight_decay = 0.0005

def block(x, width, stride, dropout, ftype='dct', sz=3):
    o1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5,  gamma_initializer='uniform')(x)
    o1 = tf.keras.layers.Activation('relu')(o1)
    y = LinearHarmonic(width,
                       ftype='ftype',
                       kernel_size=(sz, sz),
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
    z = LinearHarmonic(width,
                      ftype=ftype,
                      kernel_size=(sz, sz),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      use_bias=False)(o2)
    if z.shape[-1] != x.shape[-1]:
        side_conv = LinearHarmonic(width,
                                   ftype=ftype,
                                  kernel_size=(sz, sz),
                                  strides=(stride, stride),
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  use_bias=False)(o1)
        x = tf.keras.layers.Add()([z, side_conv])
    else:
        x = tf.keras.layers.Add()([z, x])

    return x


def group(x, n, width, stride, dropout, ftype='dct', sz=3):
    for i in range(n):
        x = block(x, width, stride if i==0 else 1, dropout, ftype, sz)
    return x

def wrn_harm(input_shape, ftype='dct', sz=3, depth=16, width=8, num_classes=10, dropout=0.3):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    x = HarmonicTransform(ftype=ftype, n=sz, strides=(1, 1, 1, 1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = HarmonicCombine(16, activation='relu')(x)

    for width, stride in zip(widths, [1, 2, 2]):
        x = group(x, n, width, stride, dropout=dropout, ftype=ftype, sz=sz)

    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           momentum=0.1,
                                           epsilon=1e-5,
                                           gamma_initializer='uniform')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    y = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation='softmax',
                                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
