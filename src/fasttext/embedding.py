# -*- coding: utf-8 -*-
import fasttext

INPUT_TXT = './../data/text8'
OUTPUT_PATH_SKIPGRAM = './skip_model'
OUTPUT_PATH_CBOW = './cbow_model'

# Learn the word representation using skipgram model
skipgram = fasttext.skipgram(INPUT_TXT, OUTPUT_PATH_SKIPGRAM, lr=0.02, dim=300, ws=5,
        epoch=1, min_count=1, neg=5, loss='ns', bucket=200000, minn=3, maxn=6,
        thread=4, t=1e-4, lr_update_rate=100)

# Get the vector of some word
print skipgram['word']
print("SKP FIN")

# Learn the word representation using cbow model
cbow = fasttext.cbow(INPUT_TXT, OUTPUT_PATH_CBOW, lr=0.02, dim=300, ws=5,
        epoch=1, min_count=1, neg=5, loss='ns', bucket=200000, minn=3, maxn=6,
        thread=4, t=1e-4, lr_update_rate=100)

# Get the vector of some word
print cbow['word']
print("CBW FIN")

# Load pre-trained skipgram model
SKIPGRAM_BIN = OUTPUT_PATH_SKIPGRAM + '.bin'
skipgram = fasttext.load_model(SKIPGRAM_BIN)
print skipgram['word']

# Load pre-trained cbow model
CBOW_BIN = OUTPUT_PATH_CBOW + '.bin'
cbow = fasttext.load_model(CBOW_BIN)
print cbow['word']
