# coding: utf-8

## Imports
from src.RNN import RecurrentLSTM
from src.Corpus import Corpus
from src.utils import preprocessing_file
import time

import keras # para Callbacks TODO: posiblemente moverlas a RecurrentLSTM en RNN.py

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
#
# Path to File

path_in = '../data/horoscopo_test_overfitting.txt'
path_out = '../data/horoscopo_test_overfitting_add_space.txt'

# Pre processing

print('\n Preprocess - Add Spaces \n')

to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''
signs_to_ignore = [i for i in to_ignore]

map_punctuation = {'¿': '<ai>',
                   '?': '<ci>',
                   '.': '<pt>',
                   '\n': '<nl>',
                   ',': '<cm>',
                   '<unk>': '<unk>',
                   ':': '<dc>',
                   ';': '<sc>'
                   }

letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'


add_space = True

if add_space:
    preprocessing_file(path_in=path_in,
                       path_out=path_out,
                       to_ignore=to_ignore
                       )

path_to_file = path_out


## Hyperparameters

D = 512 #

recurrent_dropout = 0.3 #0
dropout = 0.3 #0
dropout_seed = 0

train_size = 0.8 # 1
batch_size = 128
epochs = 300

optimizer = 'rmsprop' #'adam'
metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

workers = 1 # default 1


## Callbacks
# https://keras.io/callbacks/

out_directory_train_history = '../train_history/'
out_directory_model = '../models/'
out_model_pref = 'lstm_model_'

time_pref = time.strftime('%y%m%d.%H%M')

outfile = out_model_pref + time_pref + '.h5'

# Checkpoint
# https://keras.io/callbacks/#modelcheckpoint

monitor_checkpoint = 'val_top_k_categorical_accuracy' # 'categorical_accuracy', 'val_loss', etc


checkpoint = keras.callbacks.ModelCheckpoint(filepath=out_directory_model + outfile,
                                             monitor=monitor_checkpoint,
                                             verbose=1,
                                             save_best_only=True, # TODO: Guardar cada K epochs, y Guardar el mejor
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1
                                             )

# https://keras.io/callbacks/#earlystopping

monitor_early_stopping = 'val_top_k_categorical_accuracy' # 'categorical_accuracy','val_loss', etc

patience = 100# number of epochs with no improvement after which training will be stopped


early_stopping = keras.callbacks.EarlyStopping(monitor=monitor_early_stopping,
                                               min_delta=0,
                                               patience=patience,
                                               verbose=0,
                                               mode='auto'
                                               )

# callbacks = [checkpoint, early_stopping]
callbacks = []


##

T = 6000 # quantity of tokens

quantity_word = 30
quantity_syllable = T - quantity_word

L = 100  # sequence_length


## Init Corpus

print('\n Init Corpus \n')
corpus = Corpus(path_to_file=path_to_file,
                train_size=train_size,
                final_char=':',
                final_punc='>',
                inter_char='-',
                signs_to_ignore=signs_to_ignore,
                words_to_ignore=[],
                map_punctuation=map_punctuation,
                letters=letters,
                sign_not_syllable='<sns>'
                )


## Tokenization
print('\n Select Tokens \n')
corpus.select_tokens(quantity_word=quantity_word,
                     quantity_syllable=quantity_syllable
                     )


## L prime
print('\n L prime \n')
corpus.calculateLprime(sequence_length=L)
Lprima = corpus.lprime


## Dictionaries Token-Index
print('\n Dictionaries Token - Index \n')
corpus.dictionaries_token_index()
vocabulary = corpus.vocabulary_as_index


## Init Model
print('\n Init Model \n')
model = RecurrentLSTM(vocab_size=len(vocabulary),
                      embedding_dim=D,
                      hidden_dim=D,
                      input_length=Lprima,
                      recurrent_dropout=recurrent_dropout,
                      dropout=dropout,
                      seed=dropout_seed
                      )


## Model Summary
print('\n Model Summary \n')
print(model.summary())


## Build Model
print('\n Build Model \n')
model.build(optimizer=optimizer,
            metrics=metrics
            )


## Generators
print('\n Get Generators \n')
train_generator, eval_generator = corpus.get_generators(batch_size=batch_size)


## Training
print('\n Training \n')
ti = time.time()


model.fit(generator=train_generator,
          epochs=epochs,
          workers=workers,
          callbacks=callbacks
          )


tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))
