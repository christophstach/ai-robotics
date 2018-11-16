from os.path import realpath, dirname

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

filepath = dirname(realpath(__file__)) + '/model.h5'
print('Model: %s' % filepath)
model = load_model(filepath)

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test)

score, acc = model.evaluate(x_test, y_test)
print('score on test data: %s, accuracy on test data %s' % (score, acc))
