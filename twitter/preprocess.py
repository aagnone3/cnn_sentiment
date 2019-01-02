#Please use python 3.5 or above
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU,
                            Bidirectional, BatchNormalization, Activation, Dropout)
from keras import optimizers
from keras.models import load_model
import json, argparse, os
import re
import io
import sys
from keras.models import Model
import nltk
import string
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import num2words
import gensim
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter
)

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'l", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'em", " them", phrase)
    phrase = re.sub(r"\'nt", " not", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’l", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    phrase = re.sub(r"\’em", " them", phrase)
    phrase = re.sub(r"\’nt", " not", phrase)

    return phrase


def preprocess_twitter(dataFilePath):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
    """

    with open('../emoji/emoji_ranks.json', 'r') as fn:
        e_data = json.load(fn)

    pos_emoticons = e_data['pos']
    neg_emoticons = e_data['neg']
    neutral_emoticons = e_data['neu']

    # Emails
    emailsRegex=re.compile(r'[\w\.-]+@[\w\.-]+')

    hashtagsRegex = re.compile(r'#[a-zA-z0-9]+')
    retweetsRegex = re.compile(r'^RT.*:')

    # Mentions
    userMentionsRegex=re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z_]+[A-Za-z0-9_]+)')

    #Urls
    # urlsRegex=re.compile(r'(f|ht)(tp)(s?)(://)(.*)[.|/][^ ]+') # It may not be handling all the cases like t.co without http
    urlsRegex=re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')

    #Numerics
    numsRegex=re.compile(r"\b\d+\b")

    punctuationNotEmoticonsRegex=re.compile(r'([!?.,]){2,}')

    elongatedWords = re.compile(r'\b(\S*?)(.)\2{2,}\b')
    allCaps = re.compile(r"((?![<]*}) [A-Z][A-Z]+)")

    emoticonsDict = {}
    for i,each in enumerate(pos_emoticons):
        emoticonsDict[each]= ' <SMILE> '
    for i,each in enumerate(neg_emoticons):
        emoticonsDict[each]=' <SADFACE> '
    for i,each in enumerate(neutral_emoticons):
        emoticonsDict[each]=' <NEUTRALFACE> '
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in emoticonsDict.items())
    emoticonsPattern = re.compile("|".join(rep.keys()))
    indices = []
    elements = []
    indices = []

    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for row in finput:
            row = row.strip('\n').split('\t')
            index, line = row[0], ''.join(row[1:])


            line = urlsRegex.sub(' <URL> ', line)
            line = hashtagsRegex.sub(' <HASHTAG> ', line)
            line = retweetsRegex.sub(' <RETWEET> ', line)
            line = userMentionsRegex.sub(' <USER> ', line )
            line = emailsRegex.sub(' <EMAIL> ', line )
            line = numsRegex.sub(' <NUMBER> ',line)
            line = punctuationNotEmoticonsRegex.sub(r' \1 <REPEAT> ',line)

            line = emoticonsPattern.sub(lambda m: rep[re.escape(m.group(0))], line.strip())
            line = elongatedWords.sub(r'\1\2 <ELONG> ', line)
            line = allCaps.sub(r'\1 <ALLCAPS> ', line)
            line = re.sub('([.,!?])', r' \1 ', line)
            line = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", r" <NUMBER> ", line)
            line = re.sub(r'(.)\1{2,}', r'\1\1',line)

            line = decontracted(line.lower())
            duplicateSpacePattern = re.compile(r'\ +')
            line = re.sub(duplicateSpacePattern, ' ', line)

            indices.append(int(index))
            elements.append(line)

    return indices, elements


def get_clargs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--in-file", help="Input file.")
    parser.add_argument("-o", "--out-file", help="Output file.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_clargs()
    with open(args.out_file, 'w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(['index', 'tweet'])
        for index, tweet in zip(*preprocess_twitter(args.in_file)):
            writer.writerow([index, tweet])
