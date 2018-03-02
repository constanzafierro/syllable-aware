import unittest
import sys

class TestCorpusClass(unittest.TestCase):

    def test_init(self):

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ?'

        path = './data/test_corpusClass_init.txt'
        with open(path, 'w') as f: f.write(text)

        dict_word_true = {'bien': ['bien:'], 'como': ['co-', 'mo:'], 'estas': ['es-', 'tas:'],
                          'estás': ['es-', 'tás:'], 'hola': ['ho-', 'la:'], 'tu': ['tu:'], 'y': ['y:']}

        dict_syll_true = {'bien:': ['b-', 'i-', 'e-', 'n:'],
                          'co-': ['c-', 'o-'],
                          'es-': ['e-', 's-'],
                          'ho-': ['h-', 'o-'],
                          'la:': ['l-', 'a:'],
                          'mo:': ['m-', 'o:'],
                          'tas:': ['t-', 'a-', 's:'],
                          'tás:': ['t-', 'á-', 's:'],
                          'tu:': ['t-', 'u:'],
                          'y:': ['y:']
                          }

        freq_word_true = {'bien': 1,
                          'como': 2,
                          'estas': 1,
                          'estás': 1,
                          'hola': 2,
                          'tu': 1,
                          'y': 1}

        freq_syll_true = {'bien:': 1,
                          'co-': 2,
                          'es-': 2,
                          'ho-': 2,
                          'la:': 2,
                          'mo:': 2,
                          'tas:': 1,
                          'tás:': 1,
                          'tu:': 1,
                          'y:': 1
                          }

        train_size = 128

        final_char = ':'
        final_punc = '>'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n':'<nl>'}
        letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        sign_not_syllable = '<sns>'

        corpus = Corpus(path_to_file = path,
                        train_size = train_size,
                        final_char=final_char,
                        final_punc=final_punc,
                        inter_char=inter_char,
                        signs_to_ignore=signs_to_ignore,
                        words_to_ignore=words_to_ignore,
                        map_punctuation=map_punctuation,
                        letters=letters,
                        sign_not_syllable = sign_not_syllable
                        )

        self.assertEqual(corpus.tokenSelector.dict_word, dict_word_true)
        self.assertEqual(corpus.tokenSelector.dict_syll, dict_syll_true)
        self.assertEqual(corpus.tokenSelector.freq_word, freq_word_true)
        self.assertEqual(corpus.tokenSelector.freq_syll, freq_syll_true)


    def test_select_tokens(self):

        train_size = 128

        quantity_word = 2
        quantity_syll = 6

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ? status '
        path = './data/test_corpus_select_tokens.txt'
        with open(path, 'w') as f: f.write(text)

        final_char = ':'
        final_punc = '>'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n': '<nl>'}
        letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        sign_not_syllable = '<sns>'

        corpus = Corpus(path_to_file = path,
                        train_size = train_size,
                        final_char=final_char,
                        final_punc=final_punc,
                        inter_char=inter_char,
                        signs_to_ignore=signs_to_ignore,
                        words_to_ignore=words_to_ignore,
                        map_punctuation=map_punctuation,
                        letters=letters,
                        sign_not_syllable=sign_not_syllable
                        )

        corpus.select_tokens(quantity_word, quantity_syll)

        token_selected_true = ['hola:', '<cm>', 'como:', 'es-', 'tas:', '<nl>', 'hola:',
                               '<nl>', 'bien:', 'y:', 'tu:', 'como:', 'es-', 'tás:',
                               's-','t-', 'a-', 't-', 'u-', 's:', '<nl>']

        self.assertEqual(token_selected_true, corpus.token_selected)


    def test_calculateLprime(self):

        train_size = 128

        quantity_word = 2
        quantity_syll = 6

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ? status '
        path = './data/test_corpus_select_tokens.txt'
        with open(path, 'w') as f: f.write(text)

        final_char = ':'
        final_punc = '>'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n': '<nl>'}
        letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        sign_not_syllable = '<sns>'

        corpus = Corpus(path_to_file = path,
                        train_size = train_size,
                        final_char=final_char,
                        final_punc=final_punc,
                        inter_char=inter_char,
                        signs_to_ignore=signs_to_ignore,
                        words_to_ignore=words_to_ignore,
                        map_punctuation=map_punctuation,
                        letters=letters,
                        sign_not_syllable=sign_not_syllable
                        )

        corpus.select_tokens(quantity_word, quantity_syll)

        sequence_length = 1
        corpus.calculateLprime(sequence_length = sequence_length)

        lprime_true = 6
        self.assertEqual(corpus.lprime, lprime_true)

        sequence_length = 2
        corpus.calculateLprime(sequence_length=sequence_length)

        lprime_true = 8
        self.assertEqual(corpus.lprime, lprime_true)

        sequence_length = 3
        corpus.calculateLprime(sequence_length=sequence_length)

        lprime_true = 9
        self.assertEqual(corpus.lprime, lprime_true)

    def test_dictionaries_token_index(self):
        train_size = 128

        quantity_word = 2
        quantity_syll = 6

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ? status '
        path = './data/test_corpus_select_tokens.txt'
        with open(path, 'w') as f: f.write(text)

        final_char = ':'
        final_punc = '>'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n': '<nl>'}
        letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        sign_not_syllable = '<sns>'

        corpus = Corpus(path_to_file=path,
                        train_size=train_size,
                        final_char=final_char,
                        final_punc=final_punc,
                        inter_char=inter_char,
                        signs_to_ignore=signs_to_ignore,
                        words_to_ignore=words_to_ignore,
                        map_punctuation=map_punctuation,
                        letters=letters,
                        sign_not_syllable=sign_not_syllable
                        )

        corpus.select_tokens(quantity_word, quantity_syll)

        corpus.dictionaries_token_index()


        vocabulary_true = {'hola:', '<cm>', 'y:', 't-', 'como:', 's-', 'a-', 'tás:',
                           'tu:', 'es-', 'u-', 'bien:', 'tas:', 's:', '<nl>'}

        for key in corpus.vocabulary:
            self.assertTrue(key in vocabulary_true)


if __name__ == '__main__':

    sys.path.append("..")

    from src.Tokenization import Tokenization

    unittest.main()