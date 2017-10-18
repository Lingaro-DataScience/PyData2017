# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 14:58:14 2017

@author: dominik.lewy
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 16
epochs = 20
plot = True


def create_network(n_dense=6,
                   dense_units=16,
                   activation='selu',
                   dropout=Dropout,
                   dropout_rate=0.1,
                   kernel_initializer='RandomUniform',
                   optimizer='adam',
                   num_classes=1,
                   max_words=max_words):
    """Generic function to create a fully-connected neural network.

    # Arguments
        n_dense: int > 0. Number of dense layers.
        dense_units: int > 0. Number of dense units per layer.
        dropout: keras.layers.Layer. A dropout layer to apply.
        dropout_rate: 0 <= float <= 1. The rate of dropout.
        kernel_initializer: str. The initializer for the weights.
        optimizer: str/keras.optimizers.Optimizer. The optimizer to use.
        num_classes: int > 0. The number of classes to predict.
        max_words: int > 0. The maximum number of words per data point.

    # Returns
        A Keras model instance (compiled).
    """
    model = Sequential()
    model.add(Dense(dense_units, input_shape=(max_words,),
                    kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

    for i in range(n_dense - 1):
        model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        model.add(dropout(dropout_rate))

    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


network0 = {
    'n_dense': 2,
    'dense_units': 256,
    'activation': 'sigmoid',
    'dropout': Dropout,
    'dropout_rate': 0.5,
    'kernel_initializer': 'TruncatedNormal',
    'optimizer': 'adam'
}

network1 = {
    'n_dense': 2,
    'dense_units': 256,
    'activation': 'tanh',
    'dropout': Dropout,
    'dropout_rate': 0.5,
    'kernel_initializer': 'TruncatedNormal',
    'optimizer': 'adam'
}

network2 = {
    'n_dense': 2,
    'dense_units': 256,
    'activation': 'relu',
    'dropout': Dropout,
    'dropout_rate': 0.5,
    'kernel_initializer': 'TruncatedNormal',
    'optimizer': 'adam'
}

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('\nBuilding network 0...')

model0 = create_network(num_classes=num_classes, **network0)
history_model0 = model0.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)

score_model0 = model0.evaluate(x_test,
                               y_test,
                               batch_size=batch_size,
                               verbose=1)


print('\nBuilding network 1...')
model1 = create_network(num_classes=num_classes, **network1)
history_model1 = model1.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)

score_model1 = model1.evaluate(x_test,
                               y_test,
                               batch_size=batch_size,
                               verbose=1)


print('\nBuilding network 2...')
model2 = create_network(num_classes=num_classes, **network2)

history_model2 = model2.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)

score_model2 = model2.evaluate(x_test,
                               y_test,
                               batch_size=batch_size,
                               verbose=1)

print('\nNetwork 0 results')
print('Hyperparameters:', network0)
print('Test score:', score_model0[0])
print('Test accuracy:', score_model0[1])
print('\nNetwork 1 results')
print('Hyperparameters:', network1)
print('Test score:', score_model1[0])
print('Test accuracy:', score_model1[1])
print('Network 2 results')
print('Hyperparameters:', network2)
print('Test score:', score_model2[0])
print('Test accuracy:', score_model2[1])

plt.figure(figsize=(12,8))
plt.plot(range(epochs),
         history_model0.history['val_acc'],
         'b-',
         label='Sigmoid validation accuracy')
plt.plot(range(epochs),
         history_model1.history['val_acc'],
         'g-',
         label='Tanh validation accuracy')
plt.plot(range(epochs),
         history_model2.history['val_acc'],
         'r-',
         label='ReLU validation accuracy')
plt.plot(range(epochs),
         history_model0.history['acc'],
         'b--',
         label='Sigmoid training accuracy')
plt.plot(range(epochs),
         history_model1.history['acc'],
         'g--',
         label='Tanh training accuracy')
plt.plot(range(epochs),
         history_model2.history['acc'],
         'r--',
         label='ReLU training accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ACC_sigm_tanh_relu.png')

plt.figure(figsize=(12,8))
plt.plot(range(epochs),
         history_model0.history['val_loss'],
         'b-',
         label='Sigmoid validation lossuracy')
plt.plot(range(epochs),
         history_model1.history['val_loss'],
         'g-',
         label='Tanh validation lossuracy')
plt.plot(range(epochs),
         history_model2.history['val_loss'],
         'r-',
         label='ReLU validation lossuracy')
plt.plot(range(epochs),
         history_model0.history['loss'],
         'b--',
         label='Sigmoid training lossuracy')
plt.plot(range(epochs),
         history_model1.history['loss'],
         'g--',
         label='Tanh training lossuracy')
plt.plot(range(epochs),
         history_model2.history['loss'],
         'r--',
         label='ReLU training lossuracy')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('LOSS_sigm_tanh_relu.png')


### doesnt work
a=[]
a.append([history_model0.history['loss'],history_model0.history['val_loss'],history_model0.history['acc'],history_model0.history['val_acc']])
a.append(["tanh",history_model1.history])
a.append(["relu",history_model2.history])
import numpy as np
a
np.savetxt(fname='xxx', X=a, delimiter=',', fmt="%s")
