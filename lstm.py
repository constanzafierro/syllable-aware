# coding: utf-8

## Imports
from src.RNN import RecurrentLSTM
from src.Tokenization import Tokenization
from src.utils import preprocessing_file
from src.perplexity import metric_pp
from src.Generators import GeneralGenerator

from src.Callbacks import Callbacks


import time

from keras import backend as K


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

#path_in = './data/nicanor_clear.txt'
#path_out = './data/nicanor_clear2.txt'

#path_in = './data/train.txt'
#path_out = './data/train_add_space.txt'


##################################################

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


########################################################################################################################

## Hyperparameters

D = 512

recurrent_dropout = 0
dropout = 0

if K.backend() == 'tensorflow':
    recurrent_dropout = 0.3
    dropout = 0.3

dropout_seed = 1

train_size = 0.8 # 1
batch_size = 128
epochs = 300

optimizer = 'rmsprop' # 'adam'
metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

workers = 16 # default 1


################ CORPUS ATRIBUTES #################

T = 6000 # quantity of tokens

quantity_word = 30
quantity_syllable = T - quantity_word

L = 10  # 100 sequence_length

random_split = False
token_split = '<nl>'
use_perplexity = True


###################################################

## Init Corpus
print('\nStarting Corpus \n')
tokenization = Tokenization(path_to_file=path_to_file,
                            final_char=':',
                            final_punc='>',
                            inter_char='-',
                            signs_to_ignore=signs_to_ignore,
                            words_to_ignore=[],
                            map_punctuation=map_punctuation,
                            letters=letters,
                            sign_not_syllable='<sns>'
                            )
print('Start Corpus Done \n')


## Tokenization
print('\nSelecting Tokens \n')
tokenization.setting_tokenSelector_params(quantity_word=quantity_word,
                                          quantity_syllable=quantity_syllable
                                          )

token_selected = tokenization.select_tokens()
print('Select Tokens Done\n')

print('\nSetting experiment\n')
tokenization.setting_experiment(token_selected = token_selected, sequence_length=L)
print('Set experiment Done\n')

print("\nGet and save parameters experiment")
params_tokenization = tokenization.params_experiment()

path_setting_experiment = "./data/experimentT{}Tw{}Ts{}.txt".format(T, quantity_word, quantity_syllable)
tokenization.save_experiment(path_setting_experiment)

train_set, val_set = tokenization.split_train_val(train_size = train_size,
                                                  random_split = random_split,
                                                  token_split=token_split,
                                                  min_len = 0
                                                  )

print("size train set = {}, size val set = {}".format(len(train_set), len(val_set)))


print("average tokens per words = {}".format(params_tokenization["average_tpw"]))
if use_perplexity: metrics.append(metric_pp(average_TPW = params_tokenization["average_tpw"]))


######################## TEST COVERAGE ##################

words_cover_with_words, words_cover_with_syll, sylls_cover_with_syll = tokenization.coverage(path_to_file)
text = "With {} words the words corpus coverage is {} percent \nWith {} syllables the words corpus coverage is {} and the syllables cover is {}"
print(text.format(quantity_word,
                  words_cover_with_words,
                  quantity_syllable,
                  words_cover_with_syll,
                  sylls_cover_with_syll
                  )
      )


########################################################################################################################

## Init Model
print('\n Init Model \n')
model = RecurrentLSTM(vocab_size=len(params_tokenization["vocabulary"]),
                      embedding_dim=D,
                      hidden_dim=D,
                      input_length= params_tokenization["lprime"],
                      recurrent_dropout=recurrent_dropout,
                      dropout=dropout,
                      seed=dropout_seed
                      )


## Build Model
print('\n Build Model \n')
model.build(optimizer=optimizer,
            metrics=metrics
            )


## Model Summary
print('\n Model Summary \n')
print(model.summary)


########################################################################################################################

## Generators
print('\n Get Generators \n')

train_generator = GeneralGenerator(batch_size = batch_size,
                                   ind_tokens = train_set,
                                   vocabulary = params_tokenization["vocabulary"],
                                   max_len = params_tokenization["lprime"],
                                   split_symbol_index = token_split,
                                   count_to_split = -1,
                                   verbose = True
                                   ).__next__()

val_generator = GeneralGenerator(batch_size = batch_size,
                                 ind_tokens = val_set,
                                 vocabulary = params_tokenization["vocabulary"],
                                 max_len = params_tokenization["lprime"],
                                 split_symbol_index = token_split,
                                 count_to_split = -1
                                 ).__next__()


########################################################################################################################

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


###################################################
# Checkpoint
# https://keras.io/callbacks/#modelcheckpoint

callbacks = Callbacks()

monitor_checkpoint = 'val_top_k_categorical_accuracy' # 'val_loss'
save_best_only = True

callbacks.checkpoint(filepath=out_directory_model + outfile,
                     monitor=monitor_checkpoint,
                     save_best_only=save_best_only)


###################################################
## EarlyStopping
# https://keras.io/callbacks/#earlystopping

monitor_early_stopping = 'val_top_k_categorical_accuracy' # 'val_loss'
patience = 100 # number of epochs with no improvement after which training will be stopped

callbacks.early_stopping(monitor=monitor_early_stopping,
                         patience=patience)


###################################################
## Losswise

model_to_json = model.to_json

samples = len(train_set)
steps_per_epoch = samples / batch_size
batch_size = batch_size

callbacks.losswise(keyfile='.env',
                   model_to_json=model_to_json,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch)


###################################################

## Callbacks Pipeline
callbacks_pipeline = callbacks.get_callbacks()


########################################################################################################################

## Training
print('\n Training \n')
ti = time.time()


model.fit(train_generator=train_generator,
          val_generator=val_generator,
          epochs=epochs,
          steps_per_epoch= steps_per_epoch,
          validation_steps= len(val_set)/batch_size,
          callbacks=callbacks_pipeline,
          workers=workers,
          use_multiprocessing= True
          )


tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))


## Test
print('\nTesting\n')
######################### TEST SET ################################

path_to_test = './data/horoscopo_test_overfitting_add_space.txt'

test_set = tokenization.select_tokens(path_to_test)

index_test = tokenization.converting_token_to_index(test_set)

test_generator = GeneralGenerator(batch_size = batch_size,
                                 ind_tokens = index_test,
                                 vocabulary = params_tokenization["vocabulary"],
                                 max_len = params_tokenization["lprime"],
                                 split_symbol_index = token_split,
                                 count_to_split = -1
                                 )

path_log = './logs/'
if not os.path.exists(path=path_log):
    os.mkdir(path=path_log,
             mode=0o755
             )

path_test_result = path_log + 'test_result.txt'

ti = time.time()
scores = model.evaluate(test_generator)
tf = time.time()
dt = (tf - ti) / 60.0
print('\n Elapsed Time {} \n'.format(dt))


with open(path_test_result, "a") as f1:
    f1.write("Result experiment T = {} ; Tw = {} ; Ts = {} \nScores={}".format(T, quantity_word, quantity_syllable, scores))
