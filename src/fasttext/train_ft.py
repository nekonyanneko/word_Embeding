# -*- coding: utf-8 -*-
import fasttext as ft

# Skipgram model
model = ft.skipgram('./../data/text8', 'model')
print model.words # list of words in dictionary

'''
# CBOW model
model = ft.cbow('./../data/text8', 'model')
print model.words # list of words in dictionary
'''

# get the vector of the word 'king'
print model['king']

