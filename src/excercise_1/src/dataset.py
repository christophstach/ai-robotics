import tensorflow as tf


def cast_float(x, y):
    x = tf.cast(x, tf.float32)

    return x, y


def normalize(x, y):
    x = tf.reshape(x, (-1, 28, 28, 1))
    x = x / 255.0

    return x, y


def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x, y


def to_one_hot(x, y):
    y = tf.one_hot(y, 10)

    return x, y


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(10000)
mnist_train = mnist_train.map(cast_float)
mnist_train = mnist_train.map(normalize)
mnist_train = mnist_train.map(augment)
mnist_train = mnist_train.map(to_one_hot)
mnist_train = mnist_train.repeat()

mnist_validation = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000).shuffle(10000)
mnist_validation = mnist_validation.map(cast_float)
mnist_validation = mnist_validation.map(normalize)
mnist_validation = mnist_validation.map(to_one_hot)
mnist_validation = mnist_validation.repeat()

mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
mnist_test = mnist_test.map(cast_float)
mnist_test = mnist_test.map(normalize)
mnist_test = mnist_test.map(to_one_hot)
