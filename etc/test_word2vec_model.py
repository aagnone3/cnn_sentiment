import gensim
from gensim.models import Word2Vec

model = Word2Vec.load('word2vec.model')
print(model.most_similar("basketball"))
