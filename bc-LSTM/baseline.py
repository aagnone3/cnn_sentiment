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
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import num2words
import gensim


# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for

from preprocess import decontracted, preprocessData
from util import label2emotion, emotion2label


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix(wordIndex, out_of_vocab):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}

    # Load the embedding vectors from ther GloVe file #glove.twitter.27B.100d.txt #glove.840B.300d.txt #glove.twitter.emoji.100d.txt
    #with io.open(os.path.join(gloveDir, 'glove.twitter.emoji.100d.txt'), encoding="utf8") as f:
    vocab = []
    with io.open(gloveDir, encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
           # print(values)
            word = values[0]
            vocab.append(word)
            embeddingVector = np.array([float(val) for val in values[1:]])
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    #model = gensim.models.KeyedVectors.load_word2vec_format(gloveDir, binary=False)
    #vocab = model.vocab.keys()

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if word not in vocab:
            out_of_vocab.append(word)
#         if word in vocab:
#             embeddingVector = model[word]
#         else:
#             print(word)
#             embeddingVector = 0
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix, embeddingsIndex


def buildModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    x2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    x3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    emb1 = embeddingLayer(x1)
    emb2 = embeddingLayer(x2)
    emb3 = embeddingLayer(x3)

    lstm = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))

    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)

    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])

    inp = Reshape((3, 2*LSTM_DIM, )) (inp)

    lstm_up = LSTM(LSTM_DIM, dropout=DROPOUT)

    out_lstm = lstm_up(inp)

    out = Dense(NUM_CLASSES, activation='softmax')(out_lstm)

    adam = optimizers.adam(lr=LEARNING_RATE)
    model = Model([x1,x2,x3],out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    print(model.summary())
    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    trainDataPath = config["train_data_path"]
    valDataPath = config["val_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    trainIndices, trainTexts, labels, u1_train, u2_train, u3_train = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    writeNormalisedData(trainDataPath, trainTexts)
    print("Processing val data...")
    valIndices, valTexts, vallabels, u1_val, u2_val, u3_val = preprocessData(valDataPath, mode="train")
    print("Processing test data...")
    testIndices, testTexts, u1_test, u2_test, u3_test = preprocessData(testDataPath, mode="test")
    writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    from nltk.tokenize import TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
    print("Extracting tokens...")
    vocab = []
    for sent in u1_train+u2_train+u3_train+u1_val+u2_val+u3_val+u1_test+u2_test+u3_test:
        vocab.extend(tokenizer.tokenize(sent))

    wordIndex = {}
    for i, word in enumerate(list(set(vocab))):
        wordIndex[word] = i+1

    print(len(wordIndex))
    print(wordIndex[list(reversed(list(wordIndex)))[0]])

    u1_trainToken, u2_trainToken, u3_trainToken = [tokenizer.tokenize(x) for x in u1_train], [tokenizer.tokenize(x) for x in u2_train], [tokenizer.tokenize(x) for x in u3_train]
    u1_valToken, u2_valToken, u3_valToken = [tokenizer.tokenize(x) for x in u1_val], [tokenizer.tokenize(x) for x in u2_val], [tokenizer.tokenize(x) for x in u3_val]
    u1_testToken, u2_testToken, u3_testToken = [tokenizer.tokenize(x) for x in u1_test], [tokenizer.tokenize(x) for x in u2_test], [tokenizer.tokenize(x) for x in u3_test]

    def text_to_seq(wordIndex, tokens):
        seq = []
        for w in tokens:
            if w in wordIndex:
                seq.append(wordIndex[w])
            else:
                last_key = list(reversed(list(wordIndex)))[0]
                last_index = wordIndex[last_key]
                wordIndex[w] = last_index+1
                seq.append(wordIndex[w])
        return seq

    u1_trainSequences, u2_trainSequences, u3_trainSequences = [text_to_seq(wordIndex, x) for x in u1_trainToken], [text_to_seq(wordIndex, x) for x in u2_trainToken], [text_to_seq(wordIndex, x) for x in u3_trainToken]
    u1_valSequences, u2_valSequences, u3_valSequences = [text_to_seq(wordIndex, x) for x in u1_valToken], [text_to_seq(wordIndex, x) for x in u2_valToken], [text_to_seq(wordIndex, x) for x in u3_valToken]
    u1_testSequences, u2_testSequences, u3_testSequences = [text_to_seq(wordIndex, x) for x in u1_testToken], [text_to_seq(wordIndex, x) for x in u2_testToken], [text_to_seq(wordIndex, x) for x in u3_testToken]

    print("Found %s unique tokens." % len(wordIndex))
    print("Populating embedding matrix...")
    out_of_vocab = []
    embeddingMatrix, _ = getEmbeddingMatrix(wordIndex, out_of_vocab)
#     from sklearn.utils import class_weight
#     class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(labels),
#                                                  labels)
#     print(class_weights)
    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    u1_val = pad_sequences(u1_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_val = pad_sequences(u2_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_val = pad_sequences(u3_valSequences, maxlen=MAX_SEQUENCE_LENGTH)

    xVal = [u1_val, u2_val, u3_val]
    yVal = to_categorical(np.asarray(vallabels))

    # Randomize data
    #np.random.shuffle(trainIndices)

    #u1_data = u1_data[trainIndices]
    #u2_data = u2_data[trainIndices]
    #u3_data = u3_data[trainIndices]

    #labels = labels[trainIndices]

    # Perform k-fold cross validation
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}

    print("Starting k-fold cross validation...")
    print('-'*40)
    print("Building model...")
    model = buildModel(embeddingMatrix)
    model.fit([u1_data,u2_data,u3_data], labels,
                  validation_data=[xVal, yVal],
                  epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

    print("\n============= Metrics =================")
    print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
    print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
    print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
    print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

    print("\n======================================")

    print("Retraining model on entire data to create solution file")
    #model = buildModel(embeddingMatrix)
    #model.fit([u1_data,u2_data,u3_data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    #model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    u1_testData, u2_testData, u3_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict([u1_testData, u2_testData, u3_testData], batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
         % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))


if __name__ == '__main__':
    main()
