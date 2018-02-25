# coding: utf-8

## Imports
from src.RNN import RecurrentLSTM
from src.Corpus import Corpus
from src.utils import preprocessing_file

import time
import os

import keras # para Callbacks TODO: posiblemente moverlas a RecurrentLSTM en RNN.py

import losswise
from src.callback_losswise import LosswiseKerasCallback

########################################################################################################################

## Setting Seed for Reproducibility
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

import os
import numpy as np
import random

# import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '57' # https://github.com/fchollet/keras/issues/850
seed = 57 # must be the same as PYTHONHASHSEED
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


## Pre processing
print('\n Preprocess - Add Spaces \n')

to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&//­\xc2'''
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

D = 512

recurrent_dropout = 0
dropout = 0

if keras.backend.backend() == 'tensorflow':
    recurrent_dropout = 0.3
    dropout = 0.3

dropout_seed = 1

train_size = 0.8 # 1
batch_size = 128
epochs = 300

optimizer = 'rmsprop' # 'adam'
metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

workers = 16 # default 1


## Callbacks
# https://keras.io/callbacks/

out_directory_train_history = './train_history/'
out_directory_model = './models/'
out_model_pref = 'lstm_model_'


if not os.path.exists(path=out_directory_model):
    os.mkdir(path=out_directory_model,
             mode=0o755
             )
else:
    pass

if not os.path.exists(path=out_directory_train_history):
    os.mkdir(path=out_directory_train_history,
             mode=0o755
             )
else:
    pass


time_pref = time.strftime('%y%m%d.%H%M') # Ver código de Jorge Perez

outfile = out_model_pref + time_pref + '.h5'


# Checkpoint
# https://keras.io/callbacks/#modelcheckpoint

monitor_checkpoint = 'val_top_k_categorical_accuracy' # 'val_loss'


checkpoint = keras.callbacks.ModelCheckpoint(filepath=out_directory_model + outfile,
                                             monitor=monitor_checkpoint,
                                             verbose=1,
                                             save_best_only=True, # TODO: Guardar cada K epochs, y Guardar el mejor
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1 # Interval (number of epochs) between checkpoints.
                                             )


## EarlyStopping
# https://keras.io/callbacks/#earlystopping

monitor_early_stopping = 'val_top_k_categorical_accuracy' # 'val_loss'

patience = 100 # number of epochs with no improvement after which training will be stopped


early_stopping = keras.callbacks.EarlyStopping(monitor=monitor_early_stopping,
                                               min_delta=0,
                                               patience=patience,
                                               verbose=0,
                                               mode='auto'
                                               )
## Losswise
losswise.set_api_key('VAX1TP45Q') # api_key for "syllable-aware"
losswise_callback = LosswiseKerasCallback(tag='syllable-aware test',
                                          params_data={},
                                          params_model={})

## Callbacks Pipeline
callbacks = [checkpoint, early_stopping]#, losswise_callback]


##

T = 6000 # quantity of tokens

quantity_word = 30
quantity_syllable = T - quantity_word

L = 10  # sequence_length


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

print(model.get_config)


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
train_generator, val_generator = corpus.get_generators(batch_size=batch_size)


## Training
print('\n Training \n')
ti = time.time()


model.fit(train_generator=train_generator,
          val_generator=val_generator,
          epochs=epochs,
          callbacks=callbacks,
          workers=workers
          )


tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))


# iter = 0
# time_pref = time_pref[:-1] + str(i)