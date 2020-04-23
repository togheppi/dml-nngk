import numpy as np
from PIL import Image
import tensorflow as tf


def parse_img_fn(file_path, res=224):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)  # color images
    img = tf.image.convert_image_dtype(img, tf.float32)
    # convert unit8 tensor to floats in the [0,1]range
    img = tf.image.resize(img, [res, res])
    img = tf.image.random_flip_left_right(img)

    fn = tf.strings.split(file_path, '/')[-1]
    idx_str = tf.strings.split(fn, '.')[0]
    idx = tf.strings.to_number(idx_str, tf.dtypes.int32)

    # if idx < 10:
    #     label = tf.zeros_like(1)
    # else:
    #     label = tf.ones_like(1)

    # if tf.strings.split(file_path, '\\')[-2] == 'man':
    #     label = tf.zeros_like(1)
    # else:
    #     label = tf.ones_like(1)

    label = tf.strings.to_number(tf.strings.split(file_path, '/')[-2], tf.dtypes.int32)
    return img, label, idx, file_path


def get_dataset(img_base_dir, batch_size, epochs=None, shuffle=True):
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.list_files(img_base_dir + '/*/*', shuffle=shuffle)
        dataset = dataset.map(map_func=parse_img_fn, num_parallel_calls=8)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=40)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def test_input_fn(img_base_dir):
    batch_size = 4
    epochs = 1

    dataset = get_dataset(img_base_dir, batch_size, epochs, shuffle=False)

    for images, labels, idx, fns in dataset:
        print(images.shape, labels.numpy(), idx.numpy(), fns.numpy())
        Image.fromarray(tf.cast(images[0]*255, dtype=tf.dtypes.uint8).numpy()).save('test.png')
    return


def main():
    img_base_dir = '/mnt/vision-nas/minjae/data-sets/celeba/sort_by_id'
    test_input_fn(img_base_dir)
    return


if __name__ == '__main__':
    main()
