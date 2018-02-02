from process_text import *
from token_selectors import *


def get_array_words(corpus, selectors):
    tokens = get_processed_text(corpus, selectors)
    words = []
    for token in tokens:
        words.append(token.replace(':','').replace('-',''))
    return words


def calculateL(corpus, selectors, step_t=100):

    only_word = get_selectors(corpus, quantity_word = 1.0, quantity_syllable = 0.0)
    words_array = get_array_words(corpus, only_word)
    start_index = 0

    L = 0
    for i in range(1, len(words_array) - step_t - 1):
        words = words_array[start_index: start_index + step_t]
        token_test = get_processed_text(token_to_string(words), selectors)
        if L < len(token_test):
            L = len(token_test)
            print('Largo máximo de tokens {}'.format(L))

    return L


def test():
    step_t = 100
    raw_filename = './data/horoscopo_raw.txt'
    corpus = open(raw_filename).read().lower()
    helper_get_processed_text(raw_filename, quantity_word=0.5, quantity_syllable=0.5)
    calculateL(corpus,step_t)


def stream_lines(file_name):
    file = open(file_name)
    while True:
      line = file.readline().lower()
      if not line:
        file.close()
        break
      yield line


if __name__ == '__main__':
    test()