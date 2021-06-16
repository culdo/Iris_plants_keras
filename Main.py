from __future__ import print_function

import numpy as np
import requests
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
r = requests.get(data_url)
data = r.text
indice = 0.8

data_array = np.array([[feature for feature in item.split(',')]
             for item in str(r.text[:-1]).splitlines()])
np.random.shuffle(data_array)
for subarray in data_array:
    for i, label in enumerate(subarray):
        if label == 'Iris-setosa':
            subarray[i] = 0
        elif label == 'Iris-versicolor':
            subarray[i] = 1
        elif label == 'Iris-virginica':
            subarray[i] = 2
x_data, y_data = np.hsplit(data_array.astype('float'), (4,))
y_data = y_data.reshape(150,)

''' 
for i, label in enumerate(y_data):
    y_data[i] = 1 if label=='Iris-setosa' else 2 if label=='Iris-versicolour' else 3
'''

y_data = keras.utils.to_categorical(y_data.astype('uint8'), 3)
examples = np.array(x_data).reshape(150, 4)
x_data = x_data.T

# Feature scaling (Normalizing)
for feature in x_data:
    min_val = feature.min()
    scalar = feature.max() - min_val
    for i, val in enumerate(feature):
        feature[i] = (val - min_val) / scalar

x_data = x_data.T
(x_train, x_test), (y_train, y_test) = np.split(x_data, [int(len(x_data)*indice)]), \
                                       np.split(y_data, [int(len(y_data)*indice)])

input_shape = (4,)
batch_size = 10
epoch_num = 20

model = Sequential()
model.add(Dense(256, input_shape=input_shape))
model.add(Dropout(0.8))
model.add(Dense(256))
model.add(Dropout(0.8))
model.add(Dense(3, activation='softmax'))

model.summary()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
his = model.fit(x_train, y_train, shuffle=True,
              batch_size=batch_size, epochs=epoch_num,
              verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("loss:",score[0])
print("acc:",score[1])

