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

from process_text import *
from generators import GeneralGenerator

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model

from math import ceil
import sys

import time
import argparse


'''
Arguments


--infile:
    Input File (required)
    
    
--quantity_word:
    Fraction (or quantity) of Words to consider in the Vocabulary (default=1)
    
    
--quantity_syllable:
    Fraction (or quantity) of Syllables to be considered in the Vocabulary (default=0)


--train_size:
    Fraction of the data to consider for the Train Set (default=0.8)
    

--epochs:
    Epochs (default=300)
    
    
--batch_size:
    Batch Size (default=128)


--workers
    Maximum number of processes to spin up (default=1) [for reproducibility]
    
    
--lstm_units:
    Number of units in the LSTM layer (default=512)


--dropout:
    Dropout (default=0.3)


--recurrent_dropout:
    Recurrent dropout (default=0.3)


--learning_rate:
    Learning Rate (default=0.001)


--implementation:
    Implementation [1 or 2]. Must be 2 for GPU (default=2)

    
Example:

!python3 LSTM.py --infile 'data/horoscopo_test_overfitting.txt' --quantity_word 0.4 --quantity_syllable 0.7 --train_size 0.8 --epochs 300 --batch_size 128 --workers 1 --lstm_units 512 --dropout 0.3 --recurrent_dropout 0.3 --learning_rate 0.001 --implementation 2


Short version:

!python3 LSTM.py -i 'data/horoscopo_test_overfitting.txt' -qw 1 -qs 0 -ts 0.8 -epo 300 -bs 128 -wrk 1 -lu 512 -d 0.3 -rd 0.3 -lr 0.001 -imp 2

'''

################################################################################
## PARSER

# Embeddings

max_len = 100
embedding_dim = 300


################################################################################

parser = argparse.ArgumentParser(description='Hyperparameters')


## Input File
parser.add_argument('-i', '--infile',
                    type=argparse.FileType('r', encoding='UTF-8'),
                    required=True,
                    help='Input File')


## Vocabulario
parser.add_argument('-qw','--quantity_word',
                    type=float,
                    default=1,
                    help='Fraction (or quantity) of Words to consider in the Vocabulary (default=1)')


parser.add_argument('-qs','--quantity_syllable',
                    type=float,
                    default=0,
                    help='Fraction (or quantity) of Syllables to be considered in the Vocabulary (default=0)')

## Training
parser.add_argument('-ts','--train_size',
                    type=float,
                    default=0.8,
                    help='Fraction of the data to consider for the Train Set (default=0.8)')


parser.add_argument('-epo','--epochs',
                    default=300,
                    type=int,
                    help='Epochs (default=300)')


parser.add_argument('-bs','--batch_size',
                    type=int,
                    default=128,
                    help='Batch Size (default=128)')


parser.add_argument('-wrk','--workers',
                    type=int,
                    default=2,
                    help='Maximum number of processes to spin up (default=1) [for reproducibility]')

## Model (LSTM)
parser.add_argument('-lu','--lstm_units',
                    type=int,
                    default=512,
                    help='Number of units in the LSTM layer (default=512)')


parser.add_argument('-d','--dropout',
                    type=float,
                    default=0.3,
                    help='Dropout (default=0.3)')


parser.add_argument('-rd','--recurrent_dropout',
                    type=float,
                    default=0.3,
                    help='Recurrent dropout (default=0.3)')


parser.add_argument('-lr','--learning_rate',
                    type=float,
                    default=0.001,
                    help='Learning Rate (default=0.001)')


parser.add_argument('-imp','--implementation',
                    type=int,
                    default=2,
                    help='Implementation [1 or 2]. Must be 2 for GPU (default=2)')

##
args = parser.parse_args()

################################################################################

## Prints

print('\n'*2)

for arg in vars(args):
    if arg == 'infile':
        print( '{:30} {}'.format(arg, vars(args)[arg].name) )
    else:
        print( '{:30} {}'.format(arg, vars(args)[arg]) )

print('\n'*2)


################################################################################

# Corpus

if args.infile != None:
    path_to_file = args.infile.name
    corpus = args.infile.read().lower()


# Vocabulario

if args.quantity_word != None:
    quantity_word = args.quantity_word

if args.quantity_syllable != None:
    quantity_syllable = args.quantity_syllable


# Entrenamiento

if args.train_size != None:
    train_size = args.train_size

if args.epochs != None:
    epochs = args.epochs

if args.batch_size != None:
    batch_size = args.batch_size

if args.workers != None:
    workers = args.workers


# Modelo

if args.lstm_units != None:
    lstm_units = args.lstm_units

if args.dropout != None:
    dropout = args.dropout

if args.recurrent_dropout != None:
    recurrent_dropout = args.recurrent_dropout

if args.learning_rate != None:
    learning_rate = args.learning_rate  

if args.implementation != None:
    implementation = args.implementation

################################################################################

if train_size<0 or train_size>1:
    print('Check Train Size!')
    raise ValueError
    
if dropout<0 or dropout>1:
    print('Check Dropout!')
    raise ValueError

if recurrent_dropout<0 or recurrent_dropout>1:
    print('Check Recurrent Dropout!')
    raise ValueError
    
if epochs<0:
    print('Check Epochs!')
    raise ValueError
    
if batch_size<0:
    print('Check Batch Size!')
    raise ValueError
    
if lstm_units<0:
    print('Check LSTM Units!')
    raise ValueError

if learning_rate<0:
    print('Check Learning Rate!')
    raise ValueError

if quantity_word<0:
    print('Check Quantity Word!')
    raise VauleError
    
if quantity_syllable<0:
    print('Check Quantity Syllable!')
    raise VauleError

if implementation<1 or implementation>2:
    print('Check Implementation!')
    raise VauleError

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
    print('Training model')
    train_gen = GeneralGenerator(batch_size, ind_corpus_train, voc, max_len)
    #val_gen = GeneralGenerator(batch_size, ind_val_tokens, voc, max_len)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model_output = model.fit_generator(
        train_gen.generator(),
        train_gen.steps_per_epoch,
        epochs=epochs,
        workers=workers,
        shuffle=False
        #callbacks=[print_callback]
    )
    return model


def accuracyTest(model, string_tokens, max_len, verbose=False, *args, **kwargs):
    '''
    Requiere
    
    model : 
    max_len : 
    string_tokens : 
    verbose : opcional
    
    Retorna
    accuracy :
    '''
    
    N = len(string_tokens) - max_len - 1
    accumulated_error = 0
    x_pred = np.zeros((1, max_len))
    
    for index in range(0, N):
        start_index = index
        sentence = string_tokens[start_index: start_index + max_len]
        #print('sentence: ', ''.join(sentence))
        target = string_tokens[start_index + max_len]
        
        for t, token in enumerate(sentence):
            x_pred[0, t] = token_to_index[token]
        
        preds = model.predict(x_pred, verbose=0)[0]
        predicted = index_to_token[np.argmax(preds)+1]
        
        # Cuantificar Errores
        if target != predicted:
            accumulated_error += 1
            if verbose:
                print('ERROR i = ', index)
                print(target, predicted)
    
    accuracy = 100*(1-accumulated_error/N)
    
    print('N total :', N)
    print('Accumulated Error : ', accumulated_error)
    print('Accuracy : {0:.2f}'.format(accuracy))
    
    return accuracy

################################


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


# Accuracy Test
print('\n'*5 + 'ACCURACY TEST' + '\n'*5)
accuracyTest(model, string_tokens, max_len=max_len, verbose=False)






###########################
# Quick test of correctness:
#   select the 'horoscopo_test_overfitting.txt'
#   set True the test
###########################

max_len = 100

if False:
    # Create X, Y test
    print("---- Evaluation")
    ind_corpus_test = ind_corpus_train # using test_data = train_data to check correctness -> we should have ~1 perplexity
    num_test = ceil(len(ind_corpus_test)/(max_len+1))
    X_test = np.zeros((num_test, max_len), dtype = np.int32)
    Y_test = np.zeros((num_test, len(voc)), dtype = np.bool)
    test_count = 0
    for i in np.arange(0, len(ind_corpus_test), max_len+1):
        j = i+max_len if i+max_len < len(ind_corpus_test) else len(ind_corpus_test)-1
        pad_length = max_len-(j-i)
        for k, ind_token in enumerate([0]*pad_length + ind_corpus_test[i:j]):
            X_test[test_count, k] = ind_token
        Y_test[test_count, ind_corpus_test[j]-1] = 1
        test_count += 1
    loss = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print(model.metrics_names[0], loss)
    print("Perplexity:", np.exp(loss))

if False:
    for i in range(5):
        print('----- test ', i+1, '-----')
        start_index = random.randint(0, len(string_tokens) - max_len - 1)
        sentence = string_tokens[start_index: start_index + max_len]
        print('sentence: ', ''.join(sentence))
        print('next: ', string_tokens[start_index + max_len])
        x_pred = np.zeros((1, max_len))
        for t, token in enumerate(sentence):
            x_pred[0, t] = token_to_index[token]
        preds = model.predict(x_pred, verbose=0)[0]
        print('predicted next:', index_to_token[np.argmax(preds)+1])
