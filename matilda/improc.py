import tensorflow as tf
import numpy as np




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


