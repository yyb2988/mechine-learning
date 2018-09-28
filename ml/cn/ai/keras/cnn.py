from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets('D:\python_workspace\ml_file\MNIST_data', one_hot=True)
    # x_train = np.random.random(size=(200, 100, 100, 3))
    # y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(200, 1)))
    # x_test = np.random.random(size=(20, 100, 100, 3))
    # y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(200, 1)))

    x_train = np.reshape(a=mnist.train.images[0:10000], newshape=(-1, 28, 28, 1))
    y_train = mnist.train.labels[0:10000]
    x_val = np.reshape(a=mnist.validation.images, newshape=(-1, 28, 28, 1))
    y_val = mnist.validation.labels
    x_test = np.reshape(a=mnist.test.images, newshape=(-1, 28, 28, 1))
    y_test = mnist.test.labels

    model = Sequential()

    # input 28*28*1 get 32 个 24*24 small images
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))

    # input 32 个 24*24 small images , get 32个 12*12 small images
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # input 12*12, 10*10 small images
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Activation('relu'))

    # input 64 个 10*10 small images , 5*5 small images
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(56, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train,
              epochs=1,
              batch_size=28)

    loss_and_metrics = model.evaluate(x=x_val, y=y_val, batch_size=28)
    print(loss_and_metrics)

    total = 0
    ok = 0
    for i, x in enumerate(x_test):
        real_y_list = y_test[i]
        predict_y = model.predict(x=x_test[i:i+1])[0]
        total += 1

        pre_y = np.where(predict_y == np.max(predict_y))[0][0]
        real_y = np.where(real_y_list == np.max(real_y_list))[0][0]
        if pre_y == real_y:
            ok += 1
        print("pre_y", pre_y, ", real_y", real_y, ",", ok, "/", total)