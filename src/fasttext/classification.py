# -*- coding: utf-8 -*-
'''
export FASTTEXT_PATH="your/fasttext/path"
echo $FASTTEXT_PATH
'''

import commands as cm

# For textData input, we training by fasttext.
cmd = cm.getoutput("$FASTTEXT_PATH/fasttext supervised -input ./../data/text9 \
	-output model_Class \
	-dim 300 \
	-lr 0.1 \
	-wordNgrams 2 \
	-minCount 1 \
	-bucket 1000000 \
	-epoch 1000 \
	-thread 8")
print cmd
# For unknown words, Model output word embedding.
cmd = cm.getoutput("$FASTTEXT_PATH/fasttext test model_Class.bin ./../data/test9")
#cmd = cm.getoutput("$FASTTEXT_PATH/fasttext predict model_Class.bin ./../data/test9")
print cmd

