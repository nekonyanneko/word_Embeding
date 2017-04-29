import gensim
sentences = gensim.models.word2vec.Text8Corpus("/tmp/text8")
# size:axis
model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=1)
model.save("/tmp/text8.model")
print model["originated"]
