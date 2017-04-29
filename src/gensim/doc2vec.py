# -*- coding: utf-8 -*-

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# 空のリストを作成（学習データとなる各文書を格納）
training_docs = []

# 各文書を表すTaggedDocumentクラスのインスタンスを作成
# words：文書に含まれる単語のリスト（単語の重複あり）
# tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）
sent1 = TaggedDocument(words=['どーも', '谷口', 'です', 'よーし', 'やってく', 'ぞ'], tags=['d1'])
sent2 = TaggedDocument(words=['どーも', '谷口', 'です', 'よーし', 'がんばる', 'ぞ'], tags=['d2'])
sent3 = TaggedDocument(words=['YO', '!', 'DJ', '浅野', 'の', 'ラップ', '講座', 'DAZE', '!'], tags=['d3'])
sent4 = TaggedDocument(words=['YO', '!', 'DJ', '浅野', 'の', 'フリー', 'スタイル', 'ラップ', 'DAZE', '!'], tags=['d4'])

# 各TaggedDocumentをリストに格納
training_docs.append(sent1)
training_docs.append(sent2)
training_docs.append(sent3)
training_docs.append(sent4)

# 学習実行（パラメータを調整可能）
# documents:学習データ（TaggedDocumentのリスト）
# min_count=1:最低1回出現した単語を学習に使用する
# dm=0:学習モデル=DBOW（デフォルトはdm=1:学習モデル=DM）
model = Doc2Vec(documents=training_docs, min_count=1, dm=0)

# 学習したモデルを保存
model.save('doc2vec.model')

# 保存したモデルを読み込む場合
# model = Doc2Vec.load('doc2vec.model')

# ベクトル'd1'を表示（型はnumpy.ndarray）
# print(model.docvecs['d1'])

# 各文書と最も類似度が高い文書を表示（デフォルト値：10個）
print(model.docvecs.most_similar('d1'))
print(model.docvecs.most_similar('d2'))
print(model.docvecs.most_similar('d3'))
print(model.docvecs.most_similar('d4'))
print(model.docvecs['d1'])
print(model.docvecs['d2'])
print(model.docvecs['d3'])
print(model.docvecs['d4'])
