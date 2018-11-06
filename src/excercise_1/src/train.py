from os.path import isfile
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

filepath = 'model.h5'
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

if isfile(filepath):
    model = load_model(filepath)
else:
    model = Sequential([
        Conv2D(filters=64, kernel_size=5, input_shape=(28, 28, 1)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(filters=64, kernel_size=5),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(filters=64, kernel_size=5),
        BatchNormalization(),
        Activation('relu'),

        Flatten(),

        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(32),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(10),
        BatchNormalization(),
        Activation('softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, validation_split=0.33, epochs=10, callbacks=[
    ModelCheckpoint(filepath=filepath, save_weights_only=False, save_best_only=True)
])
