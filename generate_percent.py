from process_text import *
from generators import GeneralGenerator

# get processed text

corpus = open('./data/horoscopo_raw.txt').read().lower()[1:1000]
for a in range(11):
        quantity_syllable = a/10.0
        quantity_word = 1-a/10.0

        print('\nquantity of syllable = {} ; quantity of words = {}'.format(quantity_syllable, quantity_word))
        print('Process text...')
        selectors = get_selectors(corpus, quantity_word, quantity_syllable)
        string_tokens = get_processed_text(corpus, selectors)
        print('tokens length:', len(string_tokens))
        # crear diccionario tokens-int
        print('Vectorization...')
        string_voc = set(string_tokens)
        token_to_index = dict((t, i) for i, t in enumerate(string_voc, 1))
        index_to_token = dict((token_to_index[t], t) for t in string_voc)
        # traducir corpus a enteros
        ind_corpus = [token_to_index[token] for token in string_tokens]
        voc = set(ind_corpus)
        print('voc size:', len(voc))
        print(string_voc)