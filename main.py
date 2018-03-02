from src.RNN import RecurrentLSTM
from src.Tokenization import Tokenization
from src.utils import preprocessing_file
from src.perplexity import metric_pp
from src.Generators import GeneralGenerator
from src.Callbacks import Callbacks

import time
import os
import numpy as np
import random
import json

from keras import backend as K

os.environ['PYTHONHASHSEED'] = '1' # https://github.com/fchollet/keras/issues/850
seed = 1 # must be the same as PYTHONHASHSEED
np.random.seed(seed)
random.seed(seed)

if K.backend() == 'tensorflow':
    import tensorflow as tf

    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    tf.set_random_seed(seed)
    K.set_session(sess)

path_in = './data/train.txt'
path_out = './data/train2.txt'

out_directory_train_history = '../train_history/'
out_directory_model = '../models/'
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

if not os.path.exists(path= out_directory_model + "experiment/"):
    os.mkdir(path= out_directory_model + "experiment/",
             mode=0o755
             )
else:
    pass


def main():

    ## Preprocessing Corpus
    print('=' * 50)
    print('Preprocessing Corpus')

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

    # Agregué caracteres con acentos hacia atrás y con dos
    # puntos para las palabras en frances y alemán
    letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyzàèìòùäëïö'

    add_space = True

    if add_space:
        preprocessing_file(path_in=path_in,
                           path_out=path_out,
                           to_ignore=to_ignore
                           )

    path_to_file = path_out

    # Parameters LSTM
    D = 512

    recurrent_dropout = 0
    dropout = 0

    if K.backend() == 'tensorflow':
        recurrent_dropout = 0.3
        dropout = 0.3

    dropout_seed = 1

    train_size = 0.8  # 1
    batch_size = 512
    epochs = 100

    optimizer = 'rmsprop'  # 'adam'
    metrics = ['top_k_categorical_accuracy', 'categorical_accuracy']

    workers = 16  # default 1


    ## Hyperparameters

    train_size = 0.8

    random_split = False
    token_split = '<nl>'
    use_perplexity = False
    sequence_length = 50

    ## Init Corpus

    print('Init Corpus ...')

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

    print('Corpus Instantiated')
    print('=' * 50)
    ##

    T_W = [30, 60, 100, 300, 600, 1000, 3000, 6000]

    Tmax = 500 # Maximum number of Tokens (without considering characters)

    for tw in T_W:

        if tw > Tmax: break

        quantity_word = tw
        quantity_syllable = Tmax -tw

        ## Config .env file

        keyfile = json.load(open('.env'))
        tag = "syllable-aware " + path_to_file + " experiment T = {} ; Tw = {} ; Ts = {}"
        keyfile["losswise_tag"] = tag.format(Tmax, quantity_word, quantity_syllable)

        with open(".env", "w") as f:
            json.dump(keyfile, f)

        ## Tokenization
        print('Selecting Tokens ...')
        tokenization.setting_tokenSelector_params(quantity_word=quantity_word,
                                                  quantity_syllable=quantity_syllable
                                                  )

        print('Corpus to Process : {}'.format(path_to_file))
        print('Vocabulary Word Size = {} \nVocabulary Syllables Size = {}\nsequence length = {}'.format(
            len(tokenization.tokenSelector.words),
            len(tokenization.tokenSelector.syllables),
            sequence_length
            )
              )
        token_selected = tokenization.select_tokens()
        print('Select Tokens Done')
        print('=' * 50)

        print('Setting experiment')
        tokenization.setting_experiment(token_selected=token_selected, sequence_length=sequence_length)
        print('Set experiment Done')

        print("Get and save parameters experiment")
        params_tokenization = tokenization.params_experiment()

        target_experiment = "experimentT{}Tw{}Ts{}".format(Tmax, quantity_word, quantity_syllable)

        path_setting_experiment = out_directory_model + "experiment/" + target_experiment + "_setting_tokenize.txt"
        tokenization.save_experiment(path_setting_experiment)

        path_setting_tokenSelector = out_directory_model + "experiment/" + target_experiment + "_setting_tokenSelector.txt"
        tokenization.save_tokenSelector(path_setting_tokenSelector)

        print("average tokens per words = {}".format(params_tokenization["average_tpw"]))
        if use_perplexity: metrics.append(metric_pp(average_TPW=params_tokenization["average_tpw"]))

        print("parameter experiment saved")
        print('=' * 50)

        ##
        print('Split corpus in train and validation set')

        train_set, val_set = tokenization.split_train_val(train_size=train_size,
                                                          random_split=random_split,
                                                          token_split=token_split,
                                                          min_len=0
                                                          )

        print("size train set = {}, size val set = {}".format(len(train_set), len(val_set)))
        print('=' * 50)

        words_cover_with_words, words_cover_with_syll, sylls_cover_with_syll = tokenization.coverage(path_to_file)
        text = "With {} words the words corpus coverage is {} percent \nWith {} syllables the words corpus coverage is {} \nWith {} syllables the syllables corpus cover is {}"
        print(text.format(quantity_word,
                          words_cover_with_words,
                          quantity_syllable,
                          words_cover_with_syll,
                          quantity_syllable,
                          sylls_cover_with_syll
                          )
              )

        ## Init Model
        print('=' * 50)
        print('Init Model')
        model = RecurrentLSTM(vocab_size=len(params_tokenization["vocabulary"]),
                              embedding_dim=D,
                              hidden_dim=D,
                              input_length=params_tokenization["lprime"],
                              recurrent_dropout=recurrent_dropout,
                              dropout=dropout,
                              seed=dropout_seed
                              )

        ## Build Model
        print('Build Model ...')
        model.build(optimizer=optimizer,
                    metrics=metrics
                    )

        print('Get Generators ...')

        train_generator = GeneralGenerator(batch_size=batch_size,
                                           ind_tokens=train_set,
                                           vocabulary=params_tokenization["vocabulary"],
                                           max_len=params_tokenization["lprime"],
                                           split_symbol_index=token_split,
                                           count_to_split=-1
                                           ).__next__()

        val_generator = GeneralGenerator(batch_size=batch_size,
                                         ind_tokens=val_set,
                                         vocabulary=params_tokenization["vocabulary"],
                                         max_len=params_tokenization["lprime"],
                                         split_symbol_index=token_split,
                                         count_to_split=-1
                                         ).__next__()


        time_pref = time.strftime('%y%m%d.%H%M')  # Ver código de Jorge Perez

        outfile = out_model_pref + target_experiment + time_pref +"_{loss:.2f}_{val_loss:.2f}" + ".h5"

        callbacks = Callbacks()

        monitor_checkpoint = 'val_top_k_categorical_accuracy'  # 'val_loss'
        save_best_only = True

        callbacks.checkpoint(filepath=out_directory_model + outfile,
                             monitor=monitor_checkpoint,
                             save_best_only=save_best_only)

        monitor_early_stopping = 'val_top_k_categorical_accuracy'  # 'val_loss'
        patience = 5  # number of epochs with no improvement after which training will be stopped

        callbacks.early_stopping(monitor=monitor_early_stopping,
                                 patience=patience)

        model_to_json = model.to_json

        samples = len(train_set)
        steps_per_epoch = samples / batch_size

        callbacks.losswise(keyfile='.env',
                           model_to_json=model_to_json,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch)

        callbacks_ = callbacks.get_callbacks()


        ## Training
        print('Training ...')
        ti = time.time()

        model.fit(train_generator=train_generator,
                  val_generator=val_generator,
                  epochs=epochs,
                  steps_per_epoch= steps_per_epoch,
                  validation_steps= len(val_set)/batch_size,
                  callbacks= callbacks_,
                  workers=workers,
                  use_multiprocessing= False
                  )

        tf = time.time()
        dt = (tf - ti) / 60.0
        print('Elapsed Time {}'.format(dt))

        print("Model was trained :)\n")


if __name__ == '__main__':
    main()

