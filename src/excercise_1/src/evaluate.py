from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

filepath = 'model.h5'
model = load_model(filepath)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

score, acc = model.evaluate(x_test, y_test)
print('score on test data: %s, accuracy on test data %s' % (score, acc))