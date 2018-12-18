#Please use python 3.5 or above
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


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """

    with open('../emoji/emoji_ranks.json', 'r') as fn:
        e_data = json.load(fn)

    pos_emoticons = e_data['pos']
    neg_emoticons = e_data['neg']
    neutral_emoticons = e_data['neu']

    # Emails
    emailsRegex=re.compile(r'[\w\.-]+@[\w\.-]+')

    # Mentions
    userMentionsRegex=re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)')

    #Urls
    urlsRegex=re.compile(r'(f|ht)(tp)(s?)(://)(.*)[.|/][^ ]+') # It may not be handling all the cases like t.co without http

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
    conversations = []
    labels = []
    u1 = []
    u2 = []
    u3 = []
    indices = []

    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for row in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
#             repeatedChars = ['.', '?', '!', ',']

            items = row.strip('\n').split('\t')
            line = '\t'.join(items[1:4])
            line = emoticonsPattern.sub(lambda m: rep[re.escape(m.group(0))], line.strip())
            line = userMentionsRegex.sub(' <USER> ', line )
            line = emailsRegex.sub(' <EMAIL> ', line )
            line = urlsRegex.sub(' <URL> ', line)
            line = numsRegex.sub(' <NUMBER> ',line)
            line = punctuationNotEmoticonsRegex.sub(r' \1 <REPEAT> ',line)
            line = elongatedWords.sub(r'\1\2 <ELONG> ', line)
            line = allCaps.sub(r'\1 <ALLCAPS> ', line)
            line = re.sub('([.,!?])', r' \1 ', line)
            line = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", r" <NUMBER> ", line)
            line = re.sub(r'(.)\1{2,}', r'\1\1',line)
            line = line.strip().split('\t')
            line_0 = decontracted(line[0].lower())
            line_1 = decontracted(line[1].lower())
            line_2 = decontracted(line[2].lower())

            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[items[4]]
                labels.append(label)

            conv = ' '.join(line)

            u1.append(line_0)
            u2.append(line_1)
            u3.append(line_2)

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(items[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels, u1, u2, u3
    else:
        return indices, conversations, u1, u2, u3
