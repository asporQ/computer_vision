#For Google Colab Version
#https://colab.research.google.com/drive/16AEiFXiU6ipa5WEL6B2YwVfycrV48-mp?usp=share_link

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Create model by using sequential structure
model = Sequential()
model.add(Dense(5, input_dim=5, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Read data from file (download at https://github.com/neokarn/computer_vision/blob/master/data.csv)
#data = np.asarray([[float(num) for num in line.split(',')] for line in open('https://github.com/neokarn/computer_vision/blob/master/data.csv')])
data = pd.read_csv('https://raw.githubusercontent.com/neokarn/computer_vision/53a504e70033f8addfbf4e019f7d89195ac8a101/data.csv',header=None)
data = np.array(data)

#Train Model
x_train = data[0:100,0:5]
y_train = data[0:100,5]
y_train = to_categorical(y_train)

x_val = data[100:120,0:5]
y_val = data[100:120,5]
y_val = to_categorical(y_val)

#save model
checkpoint = ModelCheckpoint('my_model.h5',
                             verbose=1,
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only = True)

h = model.fit(x_train, y_train,
              epochs=200, batch_size=5,
              validation_data=(x_val,y_val),
              callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])


plt.show()
