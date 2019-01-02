import csv
import gensim
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter
)

stop_words = stopwords.words('english')
vocab = Counter()


def is_reserved(tok):
    return tok[0] == '<' and tok[-1] == '>'


def is_valid(tok):
    return len(tok) > 0


def train_model(in_file, out_file):
    with open(in_file, 'r') as fp:
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
        model.save(out_file)


def get_clargs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--in-file", help="Input file.")
    parser.add_argument("-o", "--out-file", help="Output file.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_clargs()
    train_model(args.in_file, args.out_file)
