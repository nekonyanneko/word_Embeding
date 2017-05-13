# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.utils.visualize_util import plot
#import matplotlib.pyplot as plt
import fasttext as ft
import re
import os

########
# parameter of fasttext
########
INPUT_TXT     = './../data/text8'
TXT_DATA_PATH = './../data/text8'
OUTPUT_PATH   = './skip_model'
#OUTPUT_PATH  = './cbow_model'

ALGORISM   = 'SKIP' #'SKIP'or'CBOW'
TRAIN_LOAD = 'LOAD' #'TRAIN'or'LOAD'

########
# parameter of CNN
########
batch_size        = 3
num_classes       = 9
epochs            = 1
data_augmentation = True
result_dir        = './'

########
# create DeepNet model Of fasttext
#######
def createModel(ALGORISM, TRAIN_LOAD, INPUT_TXT, OUTPUT_PATH,
	lr=0.02, dim=300, ws=5,epoch=1, min_count=1, neg=5, loss='ns',
	bucket=200000, minn=3, maxn=6, thread=4, t=1e-4, lr_update_rate=100):
	if TRAIN_LOAD == 'TRAIN':
		if ALGORISM == 'SKIP':
			model = ft.skipgram(INPUT_TXT, OUTPUT_PATH, lr=lr, dim=dim, ws=ws,
				epoch=epoch, min_count=min_count, neg=neg, loss=loss, bucket=bucket,
				minn=minn, maxn=maxn, thread=thread, t=t, lr_update_rate=lr_update_rate)
		else:
			model = ft.cbow(INPUT_TXT, OUTPUT_PATH, lr=lr, dim=dim, ws=ws,
                                epoch=epoch, min_count=min_count, neg=neg, loss=loss, bucket=bucket,
                                minn=minn, maxn=maxn, thread=thread, t=t, lr_update_rate=lr_update_rate)
	else:
		if ALGORISM == 'SKIP':
			# Load pre-trained skipgram model
			SKIPGRAM_BIN = OUTPUT_PATH + '.bin'
			model = ft.load_model(SKIPGRAM_BIN)
		else:
			# Load pre-trained cbow model
			CBOW_BIN = OUTPUT_PATH + '.bin'
			model = ft.load_model(CBOW_BIN)
	return model

#############
# create word Embeding
#############
def createEmbeding(model, TXT_DATA_PATH):
	f = open(TXT_DATA_PATH)
	lines = f.readlines()
	f.close()
	data = []
	for index,line in enumerate(lines):
		words = re.split(" +", line)
		line_data = []
		for num in range(len(words)):
			line_data.append(model[words[num]])
		data.append(line_data)
	return data

#############
# word Embeding Data translate input data of CNN
############
def createDataForCnn(data_num, limit_word_num, embed_num, data):
	np_data  = []
	loc_data = []
	for i in range(data_num):
		for j in range(limit_word_num):
			for z in range(embed_num):
				if(j >= len(data[i])):
					loc_data.append(0.0)
				else:
					loc_data.append(data[i][j][z])
	np_data = np.array(loc_data).reshape(data_num,limit_word_num,embed_num,1)
	return np_data

############
# main function
############
def fasttext_main():
	model = createModel(ALGORISM, TRAIN_LOAD, INPUT_TXT, OUTPUT_PATH)
	data  = createEmbeding(model, TXT_DATA_PATH)
	return data

############
# Create model of CNN
# Conv2D
#  output_filter_dim=32
#  conv_dim=3,3
#  input_shape=32,32,3 -> x,y,RGB
#  subsample=(1,1) -> default -> stride
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
############
def createModelCNN(x_train,num_classes):
	# CNN network modeling
	main_input = Input(shape=(100,300,1), dtype='float64', name='main_input')
	
	conv_1 = Conv2D(120, (3, 50), padding='same', activation='relu')(main_input)
	pool_1 = MaxPooling2D((2, 40), strides=(1, 1), padding='same')(conv_1)
	
	conv_2 = Conv2D(120, (4, 50), padding='same', activation='relu')(main_input)
        pool_2 = MaxPooling2D((2, 40), strides=(1, 1), padding='same')(conv_2)
	
	conv_3 = Conv2D(120, (5, 50), padding='same', activation='relu')(main_input)
        pool_3 = MaxPooling2D((2, 40), strides=(1, 1), padding='same')(conv_3)
	
	output = merge([pool_1, pool_2, pool_3], mode='concat', concat_axis=1)
	flatten = Flatten()(output)
	dence_1 = Dense(512, activation='relu')(flatten)
	out = Dense(1, activation='softmax')(dence_1)
	#input,output is [input1,....,inputN]
	model = Model(input=[main_input], output=[out])
	
	'''
	model_1 = Sequential()
	model_1.add(Conv2D(120, (3, 50), padding='same', input_shape=(100,300,1)))
	model_1.add(Activation('relu'))
	model_1.add(MaxPooling2D(pool_size=(2, 40)))
	#model_1.add(Dropout(0.25))
	model_1.add(Flatten())
	
	model_2 = Sequential()
        model_2.add(Conv2D(120, (4, 50), padding='same', input_shape=(100,300,1)))
        model_2.add(Activation('relu'))
        model_2.add(MaxPooling2D(pool_size=(2, 40)))
        #model_2.add(Dropout(0.25))
	model_2.add(Flatten())
	
	model_3 = Sequential()
        model_3.add(Conv2D(120, (5, 50), padding='same', input_shape=(100,300,1)))
        model_3.add(Activation('relu'))
        model_3.add(MaxPooling2D(pool_size=(2, 40)))
	#model_3.add(Dropout(0.25))
	model_3.add(Flatten())
	
	merged = Merge([model_1, model_2, model_3], mode='concat')
	
	# all conection layer
	model = Sequential()
	model.add(merged)
	#model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	'''
	return model

if __name__ == "__main__":
	data = fasttext_main()
	print('Create Embeding Data')
        print('sentence:', len(data))
        maxNum = 0
        for index in range(len(data)):
                maxNum = max(maxNum,len(data[index]))
        print('MAXwords:', maxNum)
        print('embeding:', len(data[1][1]))
        uc_id = []
        for index in range(len(data)):
                uc_id.append([index])
	y_train = np.array(uc_id)
	y_test = np.array(uc_id)
	# Stop
	# numpy data
	x_train = createDataForCnn(len(data),100,300,data)
	x_test  = createDataForCnn(len(data),100,300,data)
	print(x_train.shape,'x_train.shape')
	print(y_test.shape,'y_test.shape')
	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test  = keras.utils.to_categorical(y_test, num_classes)
	
	model = createModelCNN(x_train,num_classes)
	
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	# model summary
	model.summary()
	#plot(model, to_file='./model.png')
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
	
	#loading model
	model_file = os.path.join(result_dir, 'model.json')
	weight_file = os.path.join(result_dir, 'model.h5')
	
	with open(model_file, 'r') as fp:
		model = model_from_json(fp.read())
	model.load_weights(weight_file)
	model.summary()
	
	# evaluate
	loss, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', loss)
	print('Test acc:', acc)

