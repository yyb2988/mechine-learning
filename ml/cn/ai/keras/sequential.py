from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.keras.optimizers import SGD


def model_1():
    model = Sequential()
    model.add(Dense(80, input_dim=784))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    mnist = input_data.read_data_sets('D:\python_workspace\ml_file\MNIST_data', one_hot=True)
    print(type(mnist.train.images))
    x_train = mnist.train.images
    y_train = mnist.train.labels
    model.fit(x=x_train, y=y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x=mnist.validation.images, y=mnist.validation.labels, batch_size=128)
    print(model.metrics_names)
    print(score)


def model_2():
    model = Sequential()
    model.add(Dense(80, activation='relu', input_shape=(5,)))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    x_train = np.random.random((1000, 5))
    y_train = np.random.randint(0, 2, size=(1000, 1))
    model.fit(x=x_train, y=y_train,
              epochs=20,
              batch_size=128)


def model_3():
    model = Sequential()
    model.add(Dense(80, activation='relu', input_shape=(5,)))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.02, momentum=0.1, decay=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    x_train = np.random.random((1000, 5))
    y_train = np.random.randint(0, 10, size=(1000, 1))
    one_hot_labels = tf.keras.utils.to_categorical(y_train, num_classes=10)
    print(one_hot_labels)
    model.fit(x=x_train, y=one_hot_labels,
              epochs=20,
              batch_size=128)


if __name__ == '__main__':
    # model_1()
    model_3()
