# coding: utf-8
## Check if Train and Test Sets are passed properly 

################################################################################
## Setting Seed for Reproducibility

import os
import numpy as np
import random

import tensorflow as tf

# Setting PYTHONHASHSEED for determinism was not listed anywhere for TensorFlow,
# but apparently it is necessary for the Theano backend
# (https://github.com/fchollet/keras/issues/850).

os.environ['PYTHONHASHSEED'] = '0'
seed = 0 # must be the same as PYTHONHASHSEED

np.random.seed(seed)
random.seed(seed)

# Limit operation to 1 thread for deterministic results.

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1
                             )

from keras import backend as K

tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

################################################################################
## Imports

from process_text import *
from generators import GeneralGenerator

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model

from math import ceil
import sys

import time
import argparse


################################################################################
## Parameters

'''
Parameters


path_to_file:
    Input File (required)
    
    
quantity_word:
    Fraction (or quantity) of Words to consider in the Vocabulary (default=1)
    
    
quantity_syllable:
    Fraction (or quantity) of Syllables to be considered in the Vocabulary (default=0)


train_size:
    Fraction of the data to consider for the Train Set (default=0.8)
    

epochs:
    Epochs (default=300)
    
    
batch_size:
    Batch Size (default=128)


workers
    Maximum number of processes to spin up (default=1) [for reproducibility]
    
    
lstm_units:
    Number of units in the LSTM layer (default=512)


dropout:
    Dropout (default=0.3)


recurrent_dropout:
    Recurrent dropout (default=0.3)


learning_rate:
    Learning Rate (default=0.001)


implementation:
    Implementation [1 or 2]. Must be 2 for GPU (default=2)


# Embeddings

max_len = 100
embedding_dim = 300

'''

################################################################################


def preprocessing(*args, **kwargs):
    '''
    Requiere
    
    path_to_file : path al documento que contiene el texto sin preprocesar
    quantity_word : Porcentaje de Palabras que conforman el Vocabulario
    quantity_syllable : Porcentaje de Sílabas que conforman el Vocabulario
    train_size : Porcentaje del texto a usar para el set de entrenamiento
    
    Example:
    path_to_file = 'data/horoscopo_test_overfitting.txt'
    quantity_word = 1
    quantity_syllable = 0
    train_size = 0.8
    '''
    
    # get processed text
    print('Process text...')
    #string_tokens = get_processed_text(path_to_file, quantity_word, quantity_syllable)
    string_tokens = helper_get_processed_text(path_to_file, quantity_word, quantity_syllable)
    print('tokens length:', len(string_tokens))
    
    # crear diccionario tokens-int
    print('Vectorization...')
    string_voc = set(string_tokens)
    token_to_index = dict((t, i) for i, t in enumerate(string_voc, 1))
    index_to_token = dict((token_to_index[t], t) for t in string_voc)
    
    # translate string corpus to integers corpus
    ind_corpus = [token_to_index[token] for token in string_tokens]
    
    # testing proposes: test/train split
    len_train = int(len(ind_corpus)*train_size)
    ind_corpus_train = ind_corpus[0:len_train]
    ind_corpus_test = ind_corpus[len_train:]
    voc = set(ind_corpus)
    print('voc size:', len(voc))
    
    return string_tokens, string_voc, token_to_index, index_to_token, ind_corpus, len_train, ind_corpus_train, ind_corpus_test, voc


#def build_model(len_voc, lstm_units=128, learning_rate=0.01, max_len=100, embedding_dim=300, implementation=2, unroll=False):
#    # build the model: a single LSTM
#    print('Build model...')
#    model = Sequential()
#    model.add(Embedding(input_dim=len_voc+1, output_dim=embedding_dim, input_length=max_len, mask_zero=True))
#    model.add(LSTM(lstm_units, unroll=unroll, implementation=implementation)) #
#    model.add(Dense(len_voc))
#    model.add(Activation('softmax'))
#    optimizer = RMSprop(lr=learning_rate)
#    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#    
#    return model


def build_model(len_voc, lstm_units=512, learning_rate=0.001,
                dropout=0.3, recurrent_dropout=0.3, seed=0,
                max_len=100, embedding_dim=300,
                implementation=1, unroll=False):
    
    embedding = Embedding(input_dim=len_voc+1,
                          output_dim=embedding_dim,
                          input_length=max_len,
                          mask_zero=True)
    
    lstm_1 = LSTM(lstm_units,
                  recurrent_dropout=recurrent_dropout,
                  return_sequences=True,
                  unroll=unroll,
                  implementation=implementation)
    
    dropout = Dropout(dropout, seed=seed)
    
    lstm_2 = LSTM(lstm_units,
                  recurrent_dropout=recurrent_dropout,
                  unroll=unroll,
                  implementation=implementation)
    
    dense = Dense(len_voc, activation='softmax')
    
    model = Sequential([embedding,
                        lstm_1,
                        dropout,
                        lstm_2,
                        dense])
    
    optimizer = RMSprop(lr=learning_rate)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['top_k_categorical_accuracy', 'categorical_accuracy'])
    
    return model


def run_model(model, ind_corpus_train, voc, epochs=300, batch_size=128, max_len=100, workers=1):
    # train model
    train_gen = GeneralGenerator(batch_size, ind_corpus_train, voc, max_len)
    #val_gen = GeneralGenerator(batch_size, ind_val_tokens, voc, max_len)

    model_output = model.fit_generator(
        train_gen.generator(),
        train_gen.steps_per_epoch,
        epochs=epochs,
        workers=workers,
        shuffle=False
    )
    return model


################################################################################
## MAIN


# Argumentos para función preprocessing
args = (path_to_file, quantity_word, quantity_syllable, train_size)

# Preprocess ...
string_tokens, string_voc, token_to_index, index_to_token, ind_corpus, len_train, ind_corpus_train, ind_corpus_test, voc = preprocessing(args)

print('Tokens')
print(string_tokens)
    
print('Vocabulario')
print(string_voc)


## build model
model = build_model(len_voc=len(voc),
                    lstm_units=lstm_units,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    seed=seed,
                    max_len=max_len,
                    embedding_dim=embedding_dim,
                    implementation=implementation,
                    unroll=False)

# Model Summary
print(model.summary())


## run model
print('Training model')

t_i = time.time()

model = run_model(model,
                  ind_corpus_train,
                  voc,
                  epochs=epochs,
                  batch_size=batch_size,
                  max_len=max_len,
                  workers=workers)

t_f = time.time() - t_i
print('\n'*5 + 'Elapsed Time : ', t_f)

#print('Saving last model:', 'model_test_overfitting.h5')
#model.save('model_test_overfitting.h5')
