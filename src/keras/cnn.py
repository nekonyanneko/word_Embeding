# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#import matplotlib.pyplot as plt
import os

batch_size = 32
num_classes = 10
epochs = 1
data_augmentation = True
result_dir='./'

# cifar-10 data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, 'train samples')
print(x_test.shape, y_test.shape, 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# data scaling
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# CNN network modeling
model = Sequential()

'''
Conv2D
 output_filter_dim=32
 conv_dim=3,3
 input_shape=32,32,3 -> x,y,RGB
 subsample=(1,1) -> default -> stride
'''
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
'''
Activation -> Activation function
'''
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# all conection layer
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
model.summary()
#plot(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))

# training
model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         validation_data=(x_test, y_test),
         shuffle=True)
#plot_history(history, result_dir)

# model save
model_json = model.to_json()
with open(os.path.join(result_dir, 'model.json'), 'w') as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(result_dir, 'model.h5'))

'''
#loading model
model_file = os.path.join(result_dir, 'model.json')
weight_file = os.path.join(result_dir, 'model.h5')

with open(model_file, 'r') as fp:
    model = model_from_json(fp.read())
model.load_weights(weight_file)
model.summary()
'''

# evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)

