# coding: utf-8

## Imports
from src.RNN import RecurrentLSTM
from src.Corpus import Corpus
from src.utils import preprocessing_file
import time

########################################################################################################################

## Setting Seed for Reproducibility
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import os
import numpy as np
import random

# import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '1' # https://github.com/fchollet/keras/issues/850
seed = 1 # must be the same as PYTHONHASHSEED
np.random.seed(seed)
random.seed(seed)

## Limit operation to 1 thread for deterministic results.
# session_conf = tf.ConfigProto( intra_op_parallelism_threads = 1 , inter_op_parallelism_threads = 1 )
# from keras import backend as K
# tf.set_random_seed(seed)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

########################################################################################################################

## Path to File

path_in = './data/horoscopo_test_overfitting.txt'
path_out = './data/horoscopo_test_overfitting_add_space.txt'


print('\n Preprocess - Add Spaces \n')

to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''
signs_to_ignore = [i for i in to_ignore]

map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>', '<unk>':'<unk>', ':':'<dc>', ';':'<sc>'}
letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'

add_space = True

if add_space:
    preprocessing_file(path_in = path_in,
                       path_out = path_out,
                       to_ignore = to_ignore
                       )

path_to_file = path_out


## Hyperparameters

D = 512 #

recurrent_dropout = 0.3 #0
dropout = 0.3 #0
dropout_seed = 0

train_size = 0.8 # 1
batch_size = 128
epochs = 100 #300

optimizer = 'rmsprop' #'adam'
metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

workers = 1 # default 1
callbacks = [] # https://keras.io/callbacks/

T = 6000 # quantity of tokens

quantity_word = 30
quantity_syllable = T - quantity_word

L = 100  # sequence_length


## Init Corpus

print('\n Init Corpus \n')
corpus = Corpus(path_to_file = path_to_file,
                train_size = train_size,
                final_char = ':',
                final_punc = '>',
                inter_char = '-',
                signs_to_ignore = signs_to_ignore,
                words_to_ignore = [],
                map_punctuation = map_punctuation,
                letters = letters
                )


## Tokenization
print('\n Select Tokens \n')
corpus.select_tokens(quantity_word = quantity_word,
                     quantity_syllable = quantity_syllable
                     )


## L prime
print('\n L prime \n')
corpus.calculateLprime(sequence_length = L)
Lprima = corpus.lprime


## Dictionaries Token-Index
print('\n Dictionaries Token - Index \n')
corpus.dictionaries_token_index()
vocabulary = corpus.vocabulary_as_index


## Init Model
print('\n Init Model \n')
model = RecurrentLSTM(vocab_size = len(vocabulary),
                      embedding_dim = D,
                      hidden_dim = D,
                      input_length = Lprima,
                      recurrent_dropout = recurrent_dropout,
                      dropout = dropout,
                      seed = dropout_seed
                      )


## Model Summary
print('\n Model Summary \n')
print(model.summary())


## Build Model
print('\n Build Model \n')
model.build(optimizer = optimizer,
            metrics = metrics
            )


## Generators
print('\n Get Generators \n')
train_generator, eval_generator = corpus.get_generators(batch_size = batch_size)


## Training
print('\n Training \n')
ti = time.time()


model.fit(generator = train_generator,
          epochs = epochs,
          workers = workers,
          callbacks = callbacks
          )


tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))
