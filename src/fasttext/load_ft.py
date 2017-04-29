# -*- coding: utf-8 -*-
import fasttext as ft

model = ft.load_model('model.bin')
print model.words # list of words in dictionary
print model['king'] # get the vector of the word 'king'
