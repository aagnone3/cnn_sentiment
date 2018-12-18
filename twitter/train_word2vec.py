import csv
import gensim
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation

stop_words = stopwords.words('english')
vocab = Counter()


def is_reserved(tok):
    return tok[0] == '<' and tok[-1] == '>'


def is_valid(tok):
    return len(tok) > 0


with open('tweets_cleaned.csv', 'r') as fp:
    reader = csv.reader(fp, delimiter='\t')
    next(reader)
    tweets = list()
    for i, line in enumerate(reader):
        line = line[1].strip('\n').split()
        toks = [
            e
            for e in line
            if e not in stop_words and len(e) > 0
        ]

        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        toks = [w.translate(table) if not is_reserved(w) else w for w in toks]
        toks = list(filter(is_valid, toks))

        vocab.update(toks)
        tweets.append(toks)

    model = gensim.models.Word2Vec(tweets, workers=4, sg=1, min_count=10)
    model.save("word2vec.model")
