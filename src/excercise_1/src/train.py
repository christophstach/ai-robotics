from datetime import datetime
from os.path import dirname, realpath, isfile

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

from dataset import mnist_train, mnist_validation
from model import MNISTModelSequential

# tf.enable_eager_execution()

filepath = dirname(realpath(__file__)) + '/model.h5'
print('Model: %s' % filepath)

if isfile(filepath):
    model = load_model(filepath)
else:
    model = MNISTModelSequential()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    mnist_train,
    validation_data=mnist_validation,

    epochs=100,
    steps_per_epoch=1000,
    validation_steps=3,
    callbacks=[
        ModelCheckpoint(filepath=filepath, save_weights_only=False, save_best_only=True),
        TensorBoard(
            log_dir=dirname(realpath(__file__)) + '/logs/3-conv-4-dense-{:%Y-%m-%d---%H-%M}'.format(datetime.now()),
            write_images=True,
            write_grads=False,
            write_graph=False,
            histogram_freq=1
        )
    ])
