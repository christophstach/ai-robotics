import tensorflow as tf
import tensorflow.keras.layers as layers


class MNISTModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MNISTModel, self).__init__(name='mnist_model')

        num_filters = 64
        num_dense_neurons = 64

        self.conv_1 = layers.Conv2D(filters=num_filters, kernel_size=5)
        self.conv_2 = layers.Conv2D(filters=num_filters, kernel_size=5)
        self.conv_3 = layers.Conv2D(filters=num_filters, kernel_size=5)

        self.dense_1 = layers.Dense(num_dense_neurons)
        self.bn_1 = layers.BatchNormalization()
        self.activation_1 = layers.Activation('relu')
        self.dropout_1 = layers.Dropout(0.25)

        self.dense_2 = layers.Dense(num_dense_neurons)
        self.bn_2 = layers.BatchNormalization()
        self.activation_2 = layers.Activation('relu')
        self.dropout_2 = layers.Dropout(0.25)

        self.dense_3 = layers.Dense(num_dense_neurons)
        self.bn_3 = layers.BatchNormalization()
        self.activation_3 = layers.Activation('relu')
        self.dropout_3 = layers.Dropout(0.25)

        self.dense_4 = layers.Dense(num_classes)
        self.bn_4 = layers.BatchNormalization()
        self.activation_4 = layers.Activation('sigmoid')

    def call(self, x, training=None, mask=None):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = layers.Flatten()(x)

        x = self.dropout_1(self.activation_1(self.bn_1(self.dense_1(x))))
        x = self.dropout_2(self.activation_2(self.bn_2(self.dense_2(x))))
        x = self.dropout_3(self.activation_3(self.bn_3(self.dense_3(x))))
        x = self.activation_4(self.bn_4(self.dense_4(x)))

        return x


def MNISTModelSequential(num_classes=10):
    num_filters = 64
    num_dense_neurons = 64

    return tf.keras.models.Sequential([
        layers.Conv2D(filters=num_filters, kernel_size=5, input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(filters=num_filters, kernel_size=5),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(filters=num_filters, kernel_size=5),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Flatten(),

        layers.Dense(num_dense_neurons),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.25),

        layers.Dense(num_dense_neurons),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.25),

        layers.Dense(num_dense_neurons),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.25),

        layers.Dense(num_classes),
        layers.BatchNormalization(),
        layers.Activation('softmax')
    ])
