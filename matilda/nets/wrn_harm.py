import tensorflow as tf

from matilda.layers import Harmonic
from matilda.layers import HarmonicTransform
from matilda.layers import HarmonicCombine
from matilda.layers import LinearHarmonic

weight_decay = 0.0005

def block(x, width, stride, dropout, block_type='conv', ftype='dct', sz=3, levels=None):
    '''block types available:
       conv
       harmonic
       lin_harmonic'''
    o1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5,  gamma_initializer='uniform')(x)
    o1 = tf.keras.layers.Activation('relu')(o1)
    if block_type == 'conv':
        y = tf.keras.layers.Conv2D(width,
                     kernel_size=(sz, sz),
                     strides=(stride, stride),
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                     use_bias=False)(o1)
    elif block_type == 'harmonic':
        y = Harmonic(width,
                     ftype=ftype,
                     level = levels,
                     kernel_size=(sz, sz),
                     strides=(stride, stride),
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                     use_bias=False)(o1)
    elif block_type == 'lin_harmonic':
        y = LinearHarmonic(width,
                           ftype=ftype,
                           level = levels,
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

    if block_type == 'conv':
        z = tf.keras.layers.Conv2D(width,
                                   kernel_size=(sz, sz),
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                   use_bias=False)(o2)
    elif block_type == 'harmonic':
        z = Harmonic(width,
                     ftype=ftype,
                     level = levels,
                     kernel_size=(sz, sz),
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                     use_bias=False)(o2)
    elif block_type == 'lin_harmonic':
        z = LinearHarmonic(width,
                          ftype=ftype,
                          level = levels,
                          kernel_size=(sz, sz),
                          strides=(1, 1),
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                          use_bias=False)(o2)
    if z.shape[-1] != x.shape[-1]:
        if block_type == 'conv':
            side_conv = tf.keras.layers.Conv2D(width,
                                               kernel_size=(sz, sz),
                                               strides=(stride, stride),
                                               padding='same',
                                               kernel_initializer='he_normal',
                                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                               use_bias=False)(o1)
        elif block_type == 'harmonic':
            side_conv = Harmonic(width,
                                 ftype=ftype,
                                 level = levels,
                                 kernel_size=(sz, sz),
                                 strides=(stride, stride),
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 use_bias=False)(o1)
        elif block_type == 'lin_harmonic':
            side_conv = LinearHarmonic(width,
                                       ftype=ftype,
                                       level = levels,
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


def group(x, n, width, stride, dropout, block_type='conv', ftype='dct', sz=3, levels=None):
    for i in range(n):
        x = block(x, width, stride if i==0 else 1, dropout, block_type, ftype, sz, levels)
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
        x = group(x, n, width, stride, dropout=dropout, block_type='lin_harmonic', ftype=ftype, sz=sz, levels=None)

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

if __name__ == '__main__':
    net = wrn_harm(input_shape=(32, 32, 3),
                   ftype='chebychev',
                   sz=3,
                   depth=16,
                   width=8,
                   num_classes=10,
                   dropout=0.2)
    print(net.summary())