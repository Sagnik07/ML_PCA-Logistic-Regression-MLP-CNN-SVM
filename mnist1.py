import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mnist import MNIST
from sklearn.metrics import accuracy_score
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

arg_list = sys.argv
path = arg_list[1]

mndata = MNIST(path)
mndata.gz = True
X_train, y_train = mndata.load_training()

X_test, y_test = mndata.load_testing()
# print(len(X_test))
# print(len(y_test))
# print(len(X_test[0]))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# X_train.dtype
X_train = X_train.astype('float32')
# X_train.dtype
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose = 0)

# score = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
predictions = []
for i in range(len(y_pred)):
    predictions.append(np.argmax(y_pred[i]))

# print(predictions)
for i in predictions:
    print(i)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])