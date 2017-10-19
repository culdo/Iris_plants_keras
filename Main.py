import numpy as np
import requests
import keras
from keras.models import Sequential
from keras.layers import MaxoutDense, Dropout, Activation
import random

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
r = requests.get(data_url)
data = r.text
data_array = [int(feature) for feature in item.split(',')
              for item in str(r.text[:-1]).splitlines()]
x_data, y_data = np.hsplit(data_array,(4,))
y_data = np.array(y_data).reshape(150, 1)

for label in y_data:
    label = 1 if label = 'Iris-setosa'
    label = 2 if label = 'Iris-Versicolour'
    label = 3 if label = 'Iris-Virginica'

y_data = keras.utils.to_categorical(y_data, )

examples = np.array(x_data).reshape(30, 4)

input_shape = (1, 4)
batch_size = 10
epoch_num = 20

model = Sequential([
    MaxoutDense(128),
    Dropout(1),
    MaxoutDense(128),
    Dropout(1),
    Activation('SoftMax')
])

model.compile(optimizer='RMSprop', loss='cross_entropy', metrics='accuracy')
model.fit(x_data, y_data, shuffle=True,
          batch_size=batch_size, epochs=epoch_num,
          verbose=3, validation_split=0.2)
