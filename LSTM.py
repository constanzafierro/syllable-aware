from process_text import *

from generators import GeneralGenerator
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model

from math import ceil
import numpy as np
import random
import sys

import time


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    text = string_tokens
    start_index = random.randint(0, len(text) - max_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        sentence = text[start_index: start_index + max_len]
        generated = sentence.copy()
        print('----- Generating with seed: "' + ''.join(sentence) + '"')
        #sys.stdout.write(''.join(generated))

        for i in range(100):
            x_pred = np.zeros((1, max_len)) # no debería ser en len(sentence) ?? 
            for t, token in enumerate(sentence):
                x_pred[0, t] = token_to_index[token]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_token = index_to_token[next_index+1] # dict starting at 1

            generated += [next_token]
            sentence = sentence[1:] + [next_token]

            sys.stdout.write(next_token)
            sys.stdout.flush()
        print()


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

## Agrego unroll=True, implementation=2 a capa LSTM para ejecutarlo en google colaboratory (usando GPU)
def build_model(len_voc, max_len=100, embedding_dim=300):
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len_voc+1, output_dim=embedding_dim, input_length=max_len, mask_zero=True))
    model.add(LSTM(128, unroll=True, implementation=2)) #
    model.add(Dense(len_voc))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    return model

## Agrego workers=2 a model.fit_generator para ejecutarlo en google colaboratory (usando GPU)
def run_model(model, ind_corpus_train, voc, epochs=20, batch_size=128, max_len=100):
    # train model
    train_gen = GeneralGenerator(batch_size, ind_corpus_train, voc, max_len)
    #val_gen = GeneralGenerator(batch_size, ind_val_tokens, voc, max_len)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model_output = model.fit_generator(
        train_gen.generator(),
        train_gen.steps_per_epoch,
        epochs=epochs,
        workers=2 #,
        #callbacks=[print_callback]
    )
    return model


def accuracyTest(model, string_tokens, max_len=100, verbose=False, *args, **kwargs):
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


## MAIN


# Path al documento  = Usando doc Overfitting !!
path_to_file = 'data/horoscopo_test_overfitting.txt'


## Caso 1: Vocabulario consiste en Solamente Palabras
quantity_word = 1
quantity_syllable = 0
train_size = 0.8


# Argumentos para función preprocessing
args = (path_to_file, quantity_word, quantity_syllable, train_size)
# Preprocess ...
string_tokens, string_voc, token_to_index, index_to_token, ind_corpus, len_train, ind_corpus_train, ind_corpus_test, voc = preprocessing(args)


print('Tokens')
print(string_tokens)

print('Vocabulario')
print(string_voc)


## build model
# max_len=100, embedding_dim=300
model = build_model(len_voc= len(voc))


# Model Summary
print(model.summary())


## run model
# batch_size=128 , epochs=20
t_i = time.time()
model = run_model(model, ind_corpus, voc)
t_f = time.time() - t_i
print('\n'*5 + 'Elapsed Time : ', t_f)

#print('Saving last model:', 'model_test_overfitting.h5')
#model.save('model_test_overfitting.h5')


# Accuracy Test
print('\n'*5 + 'ACCURACY TEST' + '\n'*5)
accuracyTest(model, string_tokens, max_len=100, verbose=True)


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
