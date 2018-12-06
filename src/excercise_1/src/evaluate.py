from os.path import realpath, dirname

import tensorflow as tf

from dataset import mnist_test

filepath = dirname(realpath(__file__)) + '/model.h5'
print('Model: %s' % filepath)
model = tf.keras.models.load_model(filepath)

score, acc = model.evaluate(mnist_test, steps=10)
print('score on test data: %s, accuracy on test data %s' % (score, acc))
