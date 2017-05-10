# -*- coding: utf-8 -*-
import fasttext as ft
import re

INPUT_TXT = './../data/text8'
TXT_DATA_PATH = './../data/text8'
OUTPUT_PATH = './skip_model'
#OUTPUT_PATH = './cbow_model'

ALGORISM = 'SKIP' #'SKIP'or'CBOW'
TRAIN_LOAD = 'LOAD' #'TRAIN'or'LOAD'

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

def main():
	model = createModel(ALGORISM, TRAIN_LOAD, INPUT_TXT, OUTPUT_PATH)
	data = createEmbeding(model, TXT_DATA_PATH)
	print 'Create Embeding Data'
	print 'sentence:', len(data)
	print '   words:', len(data[1])
	print 'embeding:', len(data[1][1])

if __name__ == "__main__":
	main()

