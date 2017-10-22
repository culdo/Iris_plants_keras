import numpy as np
import requests
import keras
from keras.models import Sequential
from keras.layers import MaxoutDense, Dropout, Activation
import random

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
r = requests.get(data_url)
data = r.text

data_array = np.array([[feature for feature in item.split(',')]
             for item in str(r.text[:-1]).splitlines()])
x_data, y_data = np.hsplit(data_array, (4,))
y_data = y_data.reshape(150,)
for i, label in enumerate(y_data):
    if label == 'Iris-setosa':
        y_data[i] = 1
    elif label == 'Iris-versicolor':
        y_data[i] = 2
    else:
        y_data[i] = 3
''' 
for i, label in enumerate(y_data):
    y_data[i] = 1 if label=='Iris-setosa' else 2 if label=='Iris-versicolour' else 3
'''

x_data = float(x_data)
y_data = keras.utils.to_categorical(y_data, 3)
examples = np.array(x_data).reshape(150, 4)

input_shape = (1, 4)
batch_size = 10
epoch_num = 20

model = Sequential()
model.add(MaxoutDense(128))
model.add(Dropout(1))
model.add(MaxoutDense(128))
model.add(Dropout(1))
model.add(Activation('SoftMax'))


model.compile(optimizer='Adam', loss='cross_entropy', metrics='accuracy')
model.fit(samples_x, samples_y, shuffle=True,
          batch_size=batch_size, epochs=epoch_num,
          verbose=3, validation_spilt=0.2)

model.predict