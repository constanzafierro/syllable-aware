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

#session_conf = tf.ConfigProto( intra_op_parallelism_threads = 1 , inter_op_parallelism_threads = 1 )

#from keras import backend as K

#tf.set_random_seed(seed)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

#K.set_session(sess)


##


path_to_file = 'data/horoscopo_test_overfitting.txt'

train_size = 1

k = 1000
T = 6*k

quantity_word = 50
quantity_syllable = T - quantity_word

L = 100
Lprima = L # se debe calcular. Lprima = f(L)

D = 512

recurrent_dropout = 0.3
dropout = 0.3
dropout_seed = 0

batch_size = 128
epochs = 100

workers = 1 # default 1

callbacks = [] # https://keras.io/callbacks/



