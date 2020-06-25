import os
import numpy as np
import tensorflow as tf
from functools import partial

AUTO = tf.data.experimental.AUTOTUNE

__all__ = ['Dataset']

_datasets = {'cifar10': {'num_train': 50000,
                         'num_test': 10000,
                         'gcs_path': 'gs://cifar10_franz/',
                         'train_mask': '*train*tfrec*',
                         'test_mask': '*test*tfrec*',
                         'dimensions': [32, 32, 3]},
             'stl10': {'num_train': 5000,
                       'num_test': 8000,
                       'gcs_path': 'gs://stl10/',
                       'train_mask': '*train*tfrec*',
                       'test_mask': '*test*tfrec*',
                       'dimensions': [96, 96, 3]}}



def decode_image(image_data, imsize):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*imsize, 3])
    return image


def read_tfrecord(example, imsize):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'], imsize=imsize)
    label = tf.cast(example['label'], tf.int32)
    return image, label


def load_dataset(filenames, ordered=False, imsize=(96, 96)):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    reader = partial(read_tfrecord, imsize=imsize)

    dataset = dataset.map(reader, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    # image = tf.image.random_hue(image, 0.04)
    # image = tf.image.random_saturation(image, 0.9, 1.1)
    # image = tf.image.random_contrast(image, 0.9, 1.1)
    # image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label


def get_training_dataset(filenames, batch_size, imsize=(96, 96)):
    dataset = load_dataset(filenames, imsize)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_test_dataset(filenames, batch_size, ordered=False, imsize=(96, 96)):
    dataset = load_dataset(filenames, ordered=ordered, imsize=imsize)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


class Dataset(object):

    def __init__(self, dataset, batch_size):
        if not dataset in _datasets:
            print("Unknown dataset: {}".format(dataset))

        self._batch_size = batch_size
        self._dataset = _datasets[dataset]

    @property
    def num_train(self):
        return self._dataset['num_train']

    @property
    def num_test(self):
        return self._dataset['num_test']

    @property
    def steps_per_epoch(self):
        return self.num_train // self.batch_size

    @property
    def validation_steps(self):
        return self.num_test // self.batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def test_files(self):
        return tf.io.gfile.glob(os.path.join(self._dataset['gcs_path'],
                                             self._dataset['test_mask']))

    @property
    def train_files(self):
        return tf.io.gfile.glob(os.path.join(self._dataset['gcs_path'],
                                             self._dataset['train_mask']))

    @property
    def dims(self):
        return self._dataset['dimensions']

    def get_test(self, ordered=False):
        return get_test_dataset(self.test_files, self.batch_size, ordered=ordered, imsize=self.dims[:2])

    def get_training(self):
        return get_training_dataset(self.train_files, self.batch_size, imsize=self.dims[:2])

    def random_example(self):
        return next(iter(self.get_training().unbatch()))

def augment(example):
  image = tf.image.convert_image_dtype(example['image'], tf.float32)
  image = tf.image.resize_with_crop_or_pad(image, 108, 108)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_crop(image, size=[96, 96, 1])
  image = tf.subtract(image, np.array([125.3, 123.0, 113.9]) / 255.0)
  image = tf.divide(image, np.array([63.0, 62.1, 66.7]) / 255.0)
  return image, example['label']

def normalize(example):
  image = tf.image.convert_image_dtype(example['image'], tf.float32)
  image = tf.subtract(image, np.array([125.3, 123.0, 113.9]) / 255.0)
  image = tf.divide(image, np.array([63.0, 62.1, 66.7]) / 255.0)
  return image, example['label']

if __name__ == '__main__':
    train_ds = stl10_train.cache().shuffle(num_train_samples // 4).map(augment, num_parallel_calls=AUTO).repeat().batch(
        BATCH_SIZE).prefetch(AUTO)

