import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import nn


class DeformableConvLayer(layers.Conv2D):
    """Only support "channel last" data format"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=1,
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
        print(num_deformable_group)
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)
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
        # print("offset_num: %s" % offset_num)
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        # print(self.offset_layer_kernel)
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
        offset = nn.conv2d(inputs,
                           filters=self.offset_layer_kernel,
                           strides=[1, *self.strides, 1],
                           padding=self.padding.upper(),
                           dilations=[1, *self.dilation_rate, 1])
        # shape of offset: n_batch, feature_map_x, feature_map_y, offset_num*2
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        batch_size = K.shape(offset)[0]
        channel_in = inputs.get_shape().as_list()[-1]
        in_h, in_w = inputs.get_shape().as_list()[1:3]  # input feature map size
        out_h, out_w = offset.get_shape().as_list()[1:3]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])  # [1, out_h, out_w, filter_h*filter_w]
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]

        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in
                [y, x]]  # make batch_dim and filter_dim equal to kernel
        y, x = [tf.reshape(i, [K.shape(i)[0], *i.shape[1: 3], filter_h * filter_w * self.num_deformable_group]) for i in
                [y, x]]
        y, x = [tf.cast(i, tf.float32) for i in [y, x]]

        # print(y, x)
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
        p0, p1, p2, p3 = [self._get_pixel_values_at_point(inputs, i) for i in indices]
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
        pixels = tf.reshape(pixels,
                            [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels,
                            [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        # print("a", pixels)
        # print(channel_in)
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])
        # print("b", pixels)

        # depth-wise conv
        # print(pixels)
        # print(self.kernel)

        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')

        # add the output feature maps in the same group
        out = tf.reshape(out, [-1, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)
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

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))  # shape: 2d, x: h*w, y: h*w
        # reshape meshgrid into 1,h,w,1
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape: 4d, [1, h, w, 1]
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
        # batch = 32 if batch is None else batch
        batch = K.shape(y)[0]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

    @staticmethod
    def _rebuild_shape_to_batch(tensor_to_rebuild):
        # TODO: May I remove this??
        """In _get_pixel_values_at_point, batch dimension has already been set to stablize graph,
        We convert it back to None for batch dim
        """
        new_shape = tf.TensorShape([None]).concatenate(tensor_to_rebuild.get_shape()[1:])
        output = tf.placeholder_with_default(tensor_to_rebuild, shape=new_shape)
        return output


class DeformOffset(layers.Conv2D):
    """Only support "channel last" data format"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=1,
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
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        print(num_deformable_group)
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        # print("offset_num: %s" % offset_num)
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        # print(self.offset_layer_kernel)
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
        offset = nn.conv2d(inputs,
                           filters=self.offset_layer_kernel,
                           strides=[1, *self.strides, 1],
                           padding=self.padding.upper(),
                           dilations=[1, *self.dilation_rate, 1])
        # shape of offset: n_batch, feature_map_x, feature_map_y, offset_num*2
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        batch_size = K.shape(offset)[0]
        channel_in = inputs.get_shape().as_list()[-1]
        in_h, in_w = inputs.get_shape().as_list()[1:3]  # input feature map size
        out_h, out_w = offset.get_shape().as_list()[1:3]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])  # [1, out_h, out_w, filter_h*filter_w]
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]

        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in
                [y, x]]  # make batch_dim and filter_dim equal to kernel
        y, x = [tf.reshape(i, [K.shape(i)[0], *i.shape[1: 3], filter_h * filter_w * self.num_deformable_group]) for i in
                [y, x]]
        y, x = [tf.cast(i, tf.float32) for i in [y, x]]

        # print(y, x)
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
        p0, p1, p2, p3 = [self._get_pixel_values_at_point(inputs, i) for i in indices]
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
        # reshape the "big" feature map
        pixels = tf.reshape(pixels,
                            [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels,
                            [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        return pixels

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

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))  # shape: 2d, x: h*w, y: h*w
        # reshape meshgrid into 1,h,w,1
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape: 4d, [1, h, w, 1]
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
        # batch = 32 if batch is None else batch
        batch = K.shape(y)[0]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

    @staticmethod
    def _rebuild_shape_to_batch(tensor_to_rebuild):
        """In _get_pixel_values_at_point, batch dimension has already been set to stablize graph,
        We convert it back to None for batch dim
        """
        new_shape = tf.TensorShape([None]).concatenate(tensor_to_rebuild.get_shape()[1:])
        output = tf.placeholder_with_default(tensor_to_rebuild, shape=new_shape)
        return output



if __name__ == '__main__':
    import matilda as mt
    input = tf.keras.layers.Input(shape=(32, 32, 3))
    x = DeformableConvLayer(16, kernel_size=(4, 4), activation='relu')(input)
    m = tf.keras.models.Model(inputs=input, outputs=x)

    g = np.random.randn(1, 32, 32, 3)
    o = m(g)
    print(o.shape)

    #
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # x_train = (x_train.astype(np.float32) / 127.5) - 1
    # x_test = (x_test.astype(np.float32) / 127.5) - 1
    # y_train = tf.keras.utils.to_categorical(y_train)
    # y_test = tf.keras.utils.to_categorical(y_test)
    #
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=(32, 32, 3)),
    #     #DeformOffset(32, [4, 4], num_deformable_group=1),
    #     DeformableConvLayer(32, kernel_size=(5, 5), activation='relu'),
    #     tf.keras.layers.Conv2D(32, kernel_size=(4, 4), activation='relu'),
    #     tf.keras.layers.MaxPool2D(2, [2, 2]),
    #     tf.keras.layers.Conv2D(64, [4, 4], activation='relu'),
    #     tf.keras.layers.Conv2D(64, [4, 4], activation='relu'),
    #     tf.keras.layers.MaxPool2D(2, [2, 2]),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(10, activation='softmax')])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
    # print(max(history.history['val_accuracy']))
