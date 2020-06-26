import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
AUTO = tf.data.experimental.AUTOTUNE

class CIFAR10(object):

    def __init__(self):
        bldr = tfds.builder('cifar10')
        bldr.download_and_prepare()
        cifar10 = bldr.as_dataset(shuffle_files=True)

        self._width = 96
        self._height = 96

        self.train = cifar10['train']
        self.test = cifar10['test']
        self.num_test_samples = self._count(self.test)
        self.num_train_samples = self._count(self.train)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._width

    def get_test(self, batch_size):
        return self.test.cache().map(self._normalize, num_parallel_calls=AUTO).batch(batch_size)

    def get_train(self, batch_size):
        return self.train.cache().shuffle(self.num_train_samples//4).map(self._augment, num_parallel_calls=AUTO).repeat().batch(batch_size).prefetch(AUTO)

    def _count(self, ds):
        n = 0
        for _ in ds:
            n += 1
        return n

    def _augment(self, example):
        image = tf.image.convert_image_dtype(example['image'], tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, 38, 38)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.subtract(image, np.array([125.3, 123.0, 113.9]) / 255.0)
        image = tf.divide(image, np.array([63.0, 62.1, 66.7]) / 255.0)
        return image, example['label']

    def _normalize(self, example):
        image = tf.image.convert_image_dtype(example['image'], tf.float32)
        image = tf.subtract(image, np.array([125.3, 123.0, 113.9]) / 255.0)
        image = tf.divide(image, np.array([63.0, 62.1, 66.7]) / 255.0)
        return image, example['label']

if __name__ == '__main__':
    ds = CIFAR10()