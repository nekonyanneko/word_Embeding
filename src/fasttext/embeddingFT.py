# -*- coding: utf-8 -*-
'''
export FASTTEXT_PATH="your/fasttext/path"
echo $FASTTEXT_PATH
'''

import commands as cm

# For textData input, we training by fasttext.
cmd = cm.getoutput("$FASTTEXT_PATH/fasttext skipgram -input ./../data/text8 -output model")
print cmd
# For unknown words, Model output word embedding.
cmd = cm.getoutput("echo \"neko\" | /notebooks/fastText/fasttext print-vectors model.bin")
## cmd = cm.getoutput("echo WORD.txt | /notebooks/fastText/fasttext print-vectors model.bin")
print cmd

