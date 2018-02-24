from src.utils import preprocessing_file, Lprime
from src.Corpus import Corpus

from src.TokenSelector import TokenSelector


#TODO: ver si es útil escribir el código en función del Corpus (en vez de TokenSelector)


def main():

    ## Path to File

    path_in = '../data/horoscopo_test_overfitting.txt'
    path_out = '../data/horoscopo_test_overfitting_add_space.txt'


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


    add_space = True#False

    if add_space:
        preprocessing_file(path_in=path_in,
                           path_out=path_out,
                           to_ignore=to_ignore
                           )

    path_to_file = path_out


    ## Hyperparameters

    train_size = 1


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
                    letters=letters
                    )
    print('\n Corpus Instantiated \n')

    ##

    T_W = [30, 60, 100, 300, 600, 1000, 3000, 6000]
    Tmax = 10000 # Maximum number of Tokens (without considering characters)
    sequence_length = [100, 500, 1000, 2500, 5000]

    print('='*50)
    print('Corpus to Process : {}'.format(path_to_file))
    print('Vocabulary Word Size = {} \t Vocabulary Syllables Size = {}'.format(len(corpus.tokenSelector.dict_word),
                                                                               len(corpus.tokenSelector.dict_syll)
                                                                               )
          )


    for sl in sequence_length:

        print('='*50)
        print('sequence length = {}'.format(sl))

        for tw in T_W:

            quantity_word = tw
            quantity_syllable = Tmax - tw

            corpus.tokenSelector.get_frequent(quantity_word=quantity_word,
                                              quantity_syll=quantity_syllable
                                              )

            with open(path_out) as f1:

                for line in f1:
                    words = line.lower().split()

                    for token in words:
                        token = token.strip()

                        corpus.select_tokens(quantity_word=quantity_word,
                                             quantity_syllable=quantity_syllable
                                             )

            corpus.calculateLprime(sequence_length=sl)
            Lprima = corpus.lprime

            print('number of words = {} \t number of syllables = {} \t Lprime = {}'.format(quantity_word,
                                                                                           quantity_syllable,
                                                                                           Lprima
                                                                                           )
                  )


if __name__ == '__main__':
    main()

