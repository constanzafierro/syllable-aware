# coding: utf-8


## Setting Seed for Reproducibility

import os
import numpy as np
import random

#import tensorflow as tf

# Setting PYTHONHASHSEED for determinism was not listed anywhere for TensorFlow,
# but apparently it is necessary for the Theano backend
# (https://github.com/fchollet/keras/issues/850).

os.environ['PYTHONHASHSEED'] = '1'
seed = 1 # must be the same as PYTHONHASHSEED

np.random.seed(seed)
random.seed(seed)

# Limit operation to 1 thread for deterministic results.

# session_conf = tf.ConfigProto( intra_op_parallelism_threads = 1 , inter_op_parallelism_threads = 1 )

# from keras import backend as K

# tf.set_random_seed(seed)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

# K.set_session(sess)


## Imports

from lstmClass import Model

from corpusClass import Corpus
from corpusClass import *

import time


## Corpus

path_to_file = '../syllable-aware/data/horoscopo_test_overfitting.txt'

train_size = 1 # 0.8

corpus = Corpus(path_to_file = path_to_file,
                train_size = train_size,
                final_char = ':',
                final_punc = '>',
                inter_char = '-',
                sign_to_ignore = [],
                word_to_ignore = []
                )


## Tokenization

T = 6000 # quantity of tokens

quantity_word = 30
quantity_syllable = T - quantity_word

corpus.select_tokens(quantity_word = quantity_word,
                     quantity_syllable = quantity_syllable
                     )


## L prime

L = 100  # en el main está como sequence_length, en el mail está como L

corpus.calculateLprime(sequence_length = L)

Lprima = corpus.lprime


## LSTM Model

D = 512

recurrent_dropout = 0.3
dropout = 0.3
dropout_seed = 0

batch_size = 128
epochs = 100

workers = 1 # default 1

callbacks = [] # https://keras.io/callbacks/


## Diccionarios (vocabulario, indexaciones , etc)

corpus.dictionaries_token_index()
vocabulary = corpus.vocabulary_as_index


## Model
model = Model(vocab_size = len(vocabulary),
              embedding_dim = D,
              hidden_dim = D,
              input_length = Lprima,
              recurrent_dropout = recurrent_dropout,
              dropout = dropout,
              seed = dropout_seed
              )

print(model.summary())

optimizer = 'rmsprop' #'adam'
metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

model.build(optimizer = optimizer,
            metrics = metrics
            )


## Generators

train_generator, eval_generator = corpus.get_generators(batch_size = batch_size)


## Training

print('\n Training \n')
ti = time.time()


model.fit(generator = train_generator,
          epochs = epochs,
          workers = workers,
          callbacks = callbacks)


tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))
