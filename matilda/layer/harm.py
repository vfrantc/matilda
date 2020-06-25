import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Conv2D
import tensorflow_addons as tfa
import matplotlib.pyplot as plt


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

def dct_filters(n=3, groups=1, expand_dim=2, level=None, DC=True, l1_norm=True):
    if level is None:
        filter_bank = np.zeros((n, n, (n**2-int(not DC))), dtype=np.float32)
    else:
        filter_bank = np.zeros((n, n, (level*(level+1)//2-int(not DC))), dtype=np.float32)
    m = 0
    for i in range(n):
        for k in range(n):
            if (not DC and i == 0 and k == 0) or (not level is None and i + k >= level):
                continue
            ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
            ak = 1.0 if k > 0 else 1.0 / math.sqrt(2.0)
            for x in range(n):
                for y in range(n):
                    filter_bank[x, y, m] = math.cos((math.pi * (x + .5) * i) / n) * math.cos((math.pi * (y + .5) * k) / n)
            if l1_norm:
                filter_bank[:, :, m] /= np.sum(np.abs(filter_bank[:, :, m]))
            else:
                filter_bank[:, :, m] *= (2.0 / n) * ai * ak
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (1,1,1,groups))
    return filter_bank


class DeformableConvLayer(Conv2D):
    """Only support "channel last" data format"""
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
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
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
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
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

        self.fb = tf.Variable(initial_value=dct_filters(n=self.kernel_size[0], groups=1, expand_dim=2)[:, :, :, [0, 1, 2, 3, 4, 5,
                                                                                                                 9, 10, 11, 12, 13,
                                                                                                                 18, 19, 20, 21,
                                                                                                                 27, 28, 29,
                                                                                                                 36, 37,
                                                                                                                 45]],
                              trainable=False)



    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)

        self.harmonic_kernel = self.add_weight(shape=(1,
                                                      1,
                                                      21 * input_shape[-1],
                                                      self.filters),
                                               initializer='glorot_normal',
                                               name='combine',
                                               trainable=True)

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

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)

        self.built = True


    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h, * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        batch_size = int(inputs.get_shape()[0])
        channel_in = int(inputs.get_shape()[-1])
        in_h, in_w = [int(i) for i in inputs.get_shape()[1: 3]]  # input feature map size
        out_h, out_w = [int(i) for i in offset.get_shape()[1: 3]]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y, x = [tf.cast(i, tf.float32) for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConvLayer._get_pixel_values_at_point(inputs, i) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i, tf.float32) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, channel_in])

        # copy channels to same group
        #feat_in_group = self.filters // self.num_deformable_group
        #pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        #pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])

        groups = tf.split(pixels, axis=3, num_or_size_splits=channel_in)
        conv_groups = [tf.nn.conv2d(input=group,
                                    filters=self.fb,
                                    strides=[1, filter_h, filter_w, 1],
                                    padding='SAME') for group in groups]
        filtered = tf.concat(conv_groups, axis=3)
        out = tf.nn.conv2d(filtered, filters=self.harmonic_kernel, strides=[1, 1, 1, 1], padding='SAME')
        #
        # # at this point filtered contains stack of filter responses, for each channel
        # powered = tf.math.multiply(filtered, tf.pow(tf.abs(filtered), self.alpha))
        #
        # return tf.nn.conv2d(powered, filters=self.w, strides=(1, 1, 1, 1), padding='SAME')

        # depth-wise conv
        #out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        # add the output feature maps in the same group
        #out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        #out = tf.reduce_sum(out, axis=-1)

        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [int(i) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                         [1, *self.kernel_size, 1],
                                         [1, *self.strides, 1],
                                         [1, *self.dilation_rate, 1],
                                         'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

class ClassificationNet(keras.Model):
    def __init__(self, num_class, **kwargs):
        super().__init__(self, **kwargs)
        # classification net
        self.conv1 = DeformableConvLayer(32, [9, 9], num_deformable_group=1, activation='relu')  # out 24
        # self.conv1 = Conv2D(32, [5, 5], activation='relu')
        self.conv2 = Conv2D(32, [5, 5], activation='relu', padding='same')  # out 20
        self.max_pool1 = MaxPool2D(2, [2, 2])  # out 10
        self.conv3 = Conv2D(32, [5, 5], activation='relu', padding='same')  # out 6
        self.conv4 = Conv2D(32, [5, 5], activation='relu', padding='same')  # out 2
        self.max_pool2 = MaxPool2D(2, [2, 2])  # out 1
        self.flatten = Flatten()
        self.fc = Dense(num_class)

        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, inputs, training=None, mask=None):
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.max_pool1(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.max_pool2(net)
        net = self.flatten(net)
        logits = self.fc(net)
        return logits

    #@tf.function
    def train(self, optimizer, x, y):
        with tf.GradientTape() as tape:
            logits = self.__call__(x)
            loss = self.loss_object(y, logits)

        grads = tape.gradient(loss, self.variables)
        optimizer.apply_gradients(zip(grads, self.variables))
        return loss, tf.nn.softmax(logits)

    def accuracy(self, prediction, y):
        eq = tf.cast(tf.equal(tf.argmax(prediction, axis=-1), tf.argmax(y, axis=-1)), tf.float32)
        return tf.reduce_mean(eq)


SEED = 1234
tf.random.set_seed(SEED)


NUM_CLASS = 10
IMG_SHAPE = [28, 28]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# scale to (-1, 1), shape is (28, 28, 1)
x_train, x_test = [(np.expand_dims(i / 127.5 - 1, axis=-1)).astype(np.float32) for i in [x_train, x_test]]
y_train, y_test = tf.one_hot(y_train, depth=NUM_CLASS), tf.one_hot(y_test, depth=NUM_CLASS)


def get_dataset(batch_size, x, y, map_fn, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 10).map(map_fn, num_parallel_calls=2).batch(batch_size).prefetch(1)
    return dataset


def distorted_image_fn(image, label):
    # random rotate
    # 80% ->(-30°, 30°), 20%->(-90°,-30°)&(30°,90°)
    tf.random.set_seed(SEED)
    small_angle = tf.cast(tf.random.uniform([1], maxval=1.) <= 0.8, tf.int32)
    angle = tf.random.uniform([1], minval=0, maxval=30, dtype=tf.int32) * small_angle + \
            tf.random.uniform([1], minval=30, maxval=90, dtype=tf.int32) * (1 - small_angle)
    negative = -1 + 2 * tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32)
    angle = tf.cast(negative * angle, tf.float32)
    rotated_image = tfa.image.transform_ops.rotate(image, angle * 3.1415926 / 180)
    return rotated_image, label


def distorted_image_test_fn(image, label):
    # random rotate
    # (-135°, 135°)
    tf.random.set_seed(SEED)
    angle = tf.random.uniform([1], minval=0, maxval=135, dtype=tf.int32)
    negative = -1 + 2 * tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int32)
    angle = tf.cast(negative * angle, tf.float32)
    rotated_image = tfa.image.transform_ops.rotate(image, angle * 3.1415926 / 180)
    return rotated_image, label

if __name__ == '__main__':
    batch_size = 16

    dataset = get_dataset(batch_size, x_train, y_train, distorted_image_fn, repeat=True)
    model = ClassificationNet(num_class=NUM_CLASS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    step = 0
    for i, (rotated_image, label) in enumerate(dataset, start=1):
        step += 1
        loss, prediction = model.train(optimizer, rotated_image, label)
        acc = model.accuracy(prediction, label)

        # test
        if i % 1000 == 0:
            total_acc = 0
            dataset_test = iter(get_dataset(1000, x_test, y_test, distorted_image_test_fn))
            split = 10000 // 1000
            for _ in range(split):
                rotated_image_test, label_test = next(dataset_test)
                logits_test = model(rotated_image_test)
                prediction_test = tf.nn.softmax(logits_test)
                acc_test = model.accuracy(prediction_test, label_test).numpy()
                total_acc += acc_test
            print('test accuracy: {}'.format(total_acc / split))

        if i % 10 == 0:
            print("step: {}, loss: {}, train accuracy: {}".format(int(step), float(loss), float(acc)))