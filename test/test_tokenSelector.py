import unittest
import sys


class TestTokenSelector(unittest.TestCase):

    def test_init(self):

        final_char = ':'
        inter_char = '-'
        signs_to_ignore = ['&', '%']
        words_to_ignore = ['hola']
        map_punctuation = {'.':'<pt>', ',':'<cm>'}
        letters = 'abcdeábcdé'

        tokenselector = TokenSelector(final_char = final_char,
                                      inter_char = inter_char,
                                      signs_to_ignore = signs_to_ignore,
                                      words_to_ignore = words_to_ignore,
                                      map_punctuation = map_punctuation,
                                      letters = letters
                                      )

        self.assertEqual(tokenselector.final_char, final_char)
        self.assertEqual(tokenselector.inter_char, inter_char)
        self.assertEqual(tokenselector.signs_to_ignore, signs_to_ignore)
        self.assertEqual(tokenselector.words_to_ignore, words_to_ignore)
        self.assertEqual(tokenselector.map_punctuation, map_punctuation)
        self.assertEqual(tokenselector.punctuation, set(map_punctuation))
        self.assertEqual(tokenselector.characters, set(letters))


    def test_get_dictionary(self):

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ?'

        dict_word_true = {'bien':['bien:'], 'como':['co-', 'mo:'], 'estas':['es-', 'tas:'],
                          'estás':['es-', 'tás:'], 'hola':['ho-', 'la:'], 'tu':['tu:'], 'y':['y:']}

        dict_syll_true = {'bien:':['b-', 'i-', 'e-', 'n:'],
                          'co-':['c-', 'o-'],
                          'es-':['e-', 's-'],
                          'ho-':['h-', 'o-'],
                          'la:':['l-', 'a:'],
                          'mo:':['m-', 'o:'],
                          'tas:':['t-', 'a-', 's:'],
                          'tás:':['t-', 'á-', 's:'],
                          'tu:':['t-', 'u:'],
                          'y:': ['y:']
                          }

        freq_word_true = {'bien':1,
                          'como':2,
                          'estas':1,
                          'estás':1,
                          'hola':2,
                          'tu':1,
                          'y':1}

        freq_syll_true = {'bien:':1,
                          'co-':2,
                          'es-':2,
                          'ho-':2,
                          'la:':2,
                          'mo:':2,
                          'tas:':1,
                          'tás:':1,
                          'tu:':1,
                          'y:':1
                          }

        path = './data/test_tokenSelector_getdictionary.txt'
        with open(path, 'w') as f: f.write(text)

        final_char = ':'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n':'<nl>'}
        letters = 'abcdeábcdé'

        tokenselector = TokenSelector(final_char=final_char,
                                      inter_char=inter_char,
                                      signs_to_ignore=signs_to_ignore,
                                      words_to_ignore=words_to_ignore,
                                      map_punctuation=map_punctuation,
                                      letters=letters
                                      )

        tokenselector.get_dictionary(path)

        self.assertEqual(tokenselector.dict_word, dict_word_true)
        self.assertEqual(tokenselector.dict_syll, dict_syll_true)
        self.assertEqual(tokenselector.freq_word, freq_word_true)
        self.assertEqual(tokenselector.freq_syll, freq_syll_true)


    def test_get_frequent(self):
        quantity_word = 2
        quantity_syll = 6

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ?'
        path = './data/test_tokenSelector_getfrequent.txt'
        with open(path, 'w') as f: f.write(text)

        final_char = ':'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n': '<nl>'}
        letters = 'abcdeábcdé'

        tokenselector = TokenSelector(final_char=final_char,
                                      inter_char=inter_char,
                                      signs_to_ignore=signs_to_ignore,
                                      words_to_ignore=words_to_ignore,
                                      map_punctuation=map_punctuation,
                                      letters=letters
                                      )

        tokenselector.get_dictionary(path)

        tokenselector.get_frequent(quantity_word=quantity_word,
                                   quantity_syll=quantity_syll)

        words_selected_true = {'como', 'hola'}
        sylls_selected_true = {'es-', 'bien:', 'tas:', 'tás:', 'tu:', 'y:'}

        for w in words_selected_true:
            self.assertTrue(w in tokenselector.words)

        for s in sylls_selected_true:
            self.assertTrue(s in tokenselector.syllables)


    def test_select(self):

        quantity_word = 2
        quantity_syll = 6

        text = 'hola , ¿ como estas ? \n hola \n bien y tu como estás ?'
        path = './data/test_tokenSelector_getfrequent.txt'
        with open(path, 'w') as f: f.write(text)

        final_char = ':'
        inter_char = '-'
        signs_to_ignore = ['?', '¿']
        words_to_ignore = []
        map_punctuation = {'.': '<pt>', ',': '<cm>', '\n': '<nl>'}
        letters = 'abcdeábcdé'

        tokenselector = TokenSelector(final_char=final_char,
                                      inter_char=inter_char,
                                      signs_to_ignore=signs_to_ignore,
                                      words_to_ignore=words_to_ignore,
                                      map_punctuation=map_punctuation,
                                      letters=letters
                                      )

        tokenselector.get_dictionary(path)

        tokenselector.get_frequent(quantity_word=quantity_word,
                                   quantity_syll=quantity_syll)

        tokens = ['hola', 'como', 'estas', 'bien', 'bien', 'hola', 'tu', 'y', 'estás']
        token_selected = []

        for token in tokens:
            token_selected = tokenselector.select(token, token_selected)

        token_selected_true = ['hola:', 'como:', 'es-', 'tas:', 'bien:', 'bien:', 'hola:', 'tu:', 'y:', 'es-', 'tás']

        self.assertTrue(token_selected, token_selected_true)


if __name__ == '__main__':

    sys.path.append("..")

    from src.TokenSelector import TokenSelector

    unittest.main()