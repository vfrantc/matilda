import tensorflow as tf
import numpy as np

def gaussian_kernel(sigma=1.0):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    kernel = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[x + m, y + n] = (1 / x1) * x2

    return tf.convert_to_tensor(kernel.reshape(filter_size, filter_size, 1, 1), dtype=tf.float32)

def stuctural_tensor_eigenvalue(image):
    # compute derivatives
    derivatives = tf.image.sobel_edges(image)
    imx, imy = tf.split(derivatives, num_or_size_splits=2, axis=4)
    imx = tf.squeeze(imx, axis=4)
    imy = tf.squeeze(imy, axis=4)

    # multiply and smooth
    kernel = gaussian_kernel(sigma=0.1)
    Axx = tf.nn.depthwise_conv2d(tf.math.multiply(imx, imx), kernel, strides=(1, 1, 1, 1), padding='SAME')
    Axy = tf.nn.depthwise_conv2d(tf.math.multiply(imx, imy), kernel, strides=(1, 1, 1, 1), padding='SAME')
    Ayy = tf.nn.depthwise_conv2d(tf.math.multiply(imy, imy), kernel, strides=(1, 1, 1, 1), padding='SAME')

    # compute first eigen-value
    l1 = (Axx + Ayy) / 2 + tf.sqrt(4 * Axy ** 2 + (Axx - Ayy) ** 2) / 2
    return l1

