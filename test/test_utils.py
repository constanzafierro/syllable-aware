import unittest
import sys


class TestUtils(unittest.TestCase):

    def test_silabas(self):

        word = 'palabra'
        syll_true = 'pa-la-bra'
        answer = silabas(word, sep='-')

        self.assertEqual(syll_true, answer)


        word = 'paralelepipedo'
        syll_true = 'pa-ra-le-le-pi-pe-do'
        answer = silabas(word, sep='-')

        self.assertEqual(syll_true, answer)


        word = 'atención'
        syll_true = 'a-ten-ción'
        answer = silabas(word, sep='-')

        self.assertEqual(syll_true, answer)


        word = 'atracción'
        syll_true = 'a-trac-ción'
        answer = silabas(word, sep='-')

        self.assertEqual(syll_true, answer)


        word = 'Ññ'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')

        self.assertTrue('Estructura de sílaba incorrecta en la palabra' in str(context.exception))


        word = 'specific'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')

        self.assertTrue('Estructura de sílaba incorrecta en la palabra' in str(context.exception))


        word = 'status'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')

        self.assertTrue('Estructura de sílaba incorrecta en la palabra' in str(context.exception))


        word = ' '
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')

        self.assertTrue('No se reconoce el carácter' in str(context.exception))


        word = '-'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')

        self.assertTrue('No se reconoce el carácter' in str(context.exception))


    def test_preprocessing(self):

        text = 'hola.' + '\ncómo estás?' + '\ntan helado que estai Juan!'
        s = 'hola' + ' . ' + '\ncómo estás' + ' ? ' + '\ntan helado que estai Juan' + ' ' * 2 + '\n'
        to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

        path = './data/test_utils_preprocessing.txt'
        with open(path, 'w') as f: f.write(text)

        path_in = './data/test_utils_preprocessing.txt'
        path_out = './data/test_utils_preprocessing_spaces.txt'

        preprocessing_file(path_in=path_in,
                           path_out=path_out,
                           to_ignore=to_ignore
                           )

        file = open(path_out, 'r')
        processed = file.read()

        self.assertEqual(s, processed)

        path_in = './data/None.txt'
        path_out = './data/test_utils_preprocessing_spaces.txt'

        to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''


        with self.assertRaises(TypeError) as context:
            preprocessing_file(path_in=path_in,
                               path_out=path_out,
                               to_ignore=to_ignore
                               )
        self.assertTrue('File not exists' in str(context.exception))
        file.close()


    def test_get_syllable(self):

        word = 'palabra'
        syll_true = ['pa-', 'la-', 'bra:']
        answer = get_syllables(word=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(syll_true, answer)


        word = 'paralelepipedo'
        syll_true = ['pa-','ra-','le-','le-','pi-','pe-','do:']
        answer = get_syllables(word=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(syll_true, answer)


        word = 'atención'
        syll_true = ['a-','ten-','ción:']
        answer = get_syllables(word=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(syll_true, answer)


        word = 'atracción'
        syll_true = ['a-','trac-','ción:']
        answer = get_syllables(word=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(syll_true, answer)


    def test_get_characters(self):

        word = 'palabra:'
        char_true = ['p-', 'a-', 'l-', 'a-', 'b-', 'r-', 'a:']
        answer = get_characters(token=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


        word = 'paralelepipedo:'
        char_true = ['p-', 'a-','r-', 'a-','l-', 'e-','l-', 'e-','p-', 'i-','p-', 'e-','d-', 'o:']
        answer = get_characters(token=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


        word = 'atención:'
        char_true = ['a-','t-', 'e-', 'n-','c-', 'i-', 'ó-', 'n:']
        answer = get_characters(token=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


        word = 'atracción:'
        char_true = ['a-','t-', 'r-', 'a-', 'c-', 'c-', 'i-', 'ó-', 'n:']
        answer = get_characters(token=word,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


        syll = 'ca-'
        char_true = ['c-', 'a-']
        answer = get_characters(token=syll,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


        syll = 'pre-'
        char_true = ['p-', 'r-', 'e-']
        answer = get_characters(token=syll,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


        syll = 'post:'
        char_true = ['p-', 'o-', 's-', 't:']
        answer = get_characters(token=syll,
                               middle='-',
                               end=':'
                               )

        self.assertEqual(char_true, answer)


    def test_get_freq_words(self):

        freq_word = dict()
        word1 = 'hola'

        freq_word = get_freq_words(word1, freq_word, to_ignore=[])
        self.assertEqual(freq_word, {word1:1})

        freq_word = get_freq_words(word1, freq_word, to_ignore=[])
        self.assertEqual(freq_word, {word1:2})

        freq_word = get_freq_words(word1, freq_word, to_ignore=[])
        self.assertEqual(freq_word, {word1:3})


        word2 = 'como'

        freq_word = get_freq_words(word2, freq_word, to_ignore=[])
        self.assertEqual(freq_word, {word1:3, word2:1})

        freq_word = get_freq_words(word2, freq_word, to_ignore=[])
        self.assertEqual(freq_word, {word1:3, word2:2})

        freq_word = get_freq_words(word2, freq_word, to_ignore=[])
        self.assertEqual(freq_word, {word1:3, word2:3})


    def test_get_freq_syllables(self):

        freq_word = {'hola':5, 'la':2, 'como':3}
        dict_word = {'hola':['ho-','la:'], 'la':['la:'], 'como':['co-','mo:']}

        freq_syll = get_freq_syllables(freq_word, dict_word, to_ignore=[])
        freq_syll_true = {'co-':3, 'ho-':5, 'la:':7, 'mo:':3}

        self.assertEqual(freq_syll, freq_syll_true)


        freq_word = {'hola':5, 'la':2, 'como':3, 'amo':10}
        dict_word = {'hola':['ho-','la:'], 'la':['la:'], 'como':['co-','mo:'], 'amo':['a-', 'mo:']}

        freq_syll = get_freq_syllables(freq_word, dict_word, to_ignore=[])
        freq_syll_true = {'a-':10,'co-':3, 'ho-':5, 'la:':7, 'mo:':13}

        self.assertEqual(freq_syll, freq_syll_true)


    def test_word_to_syll(self):

        dict_word = dict()
        word = 'hola'
        dict_word = word_to_syll(word, dict_word,
                                 to_ignore=[],
                                 middle='-',
                                 end=':',
                                 sign_not_syllable='<sns>',
                                 verbose=False
                                 )
        dict_word_true = {'hola':['ho-','la:']}

        self.assertEqual(dict_word, dict_word_true)


        word = 'la'
        dict_word = word_to_syll(word, dict_word,
                                 to_ignore=[],
                                 middle='-',
                                 end=':',
                                 sign_not_syllable='<sns>',
                                 verbose=False
                                 )
        dict_word_true = {'hola':['ho-','la:'], 'la':['la:']}

        self.assertEqual(dict_word, dict_word_true)


        word = 'hola'
        dict_word = word_to_syll(word, dict_word,
                                 to_ignore=[],
                                 middle='-',
                                 end=':',
                                 sign_not_syllable='<sns>',
                                 verbose=False
                                 )
        dict_word_true = {'hola':['ho-','la:'], 'la':['la:']}

        self.assertEqual(dict_word, dict_word_true)


        word = 'como'
        dict_word = word_to_syll(word, dict_word,
                                 to_ignore=[],
                                 middle='-',
                                 end=':',
                                 sign_not_syllable='<sns>',
                                 verbose=False
                                 )
        dict_word_true = {'como':['co-','mo:'], 'hola':['ho-','la:'], 'la':['la:']}

        self.assertEqual(dict_word, dict_word_true)


        word = 'status'
        dict_word = word_to_syll(word, dict_word,
                                 to_ignore=[],
                                 middle='-',
                                 end=':',
                                 sign_not_syllable='<sns>',
                                 verbose=False
                                 )
        dict_word_true = {'como':['co-','mo:'], 'hola':['ho-','la:'], 'la':['la:'], 'status':['<sns>']}

        self.assertEqual(dict_word, dict_word_true)


        word = 'palindromo'
        to_ignore = [word]
        dict_word = word_to_syll(word, dict_word,
                                 to_ignore=to_ignore,
                                 middle='-',
                                 end=':',
                                 sign_not_syllable='<sns>',
                                 verbose=False
                                 )
        dict_word_true = {'como':['co-','mo:'], 'hola':['ho-','la:'], 'la':['la:'], 'status':['<sns>']}

        self.assertEqual(dict_word, dict_word_true)


    def test_syll_to_charac(self):

        word = 'hola'
        dict_word = {'como': ['co-', 'mo:'], 'hola': ['ho-', 'la:'], 'la': ['la:'], 'status': ['<sns>']}
        dict_syll = dict()
        dict_syll = syll_to_charac(word,
                                   dict_syll,
                                   dict_word,
                                   to_ignore=[],
                                   middle='-',
                                   end=':',
                                   sign_not_syllable='<sns>'
                                   )
        dict_syll_true = {'ho-':['h-', 'o-'], 'la:':['l-', 'a:']}

        self.assertEqual(dict_syll, dict_syll_true)


        word = 'la'
        dict_syll = syll_to_charac(word,
                                   dict_syll,
                                   dict_word,
                                   to_ignore=[],
                                   middle='-',
                                   end=':',
                                   sign_not_syllable='<sns>'
                                   )
        dict_syll_true = {'ho-': ['h-', 'o-'], 'la:': ['l-', 'a:']}

        self.assertEqual(dict_syll, dict_syll_true)


        word = 'status'
        dict_syll = syll_to_charac(word,
                                   dict_syll,
                                   dict_word,
                                   to_ignore=[],
                                   middle='-',
                                   end=':',
                                   sign_not_syllable='<sns>'
                                   )
        dict_syll_true = {'ho-': ['h-', 'o-'], 'la:': ['l-', 'a:'], 'status':['s-', 't-', 'a-', 't-', 'u-', 's:']}

        self.assertEqual(dict_syll, dict_syll_true)


        word = 'casa'
        to_ignore = [word]
        dict_syll = syll_to_charac(word,
                                   dict_syll,
                                   dict_word,
                                   to_ignore=to_ignore,
                                   middle='-',
                                   end=':',
                                   sign_not_syllable='<sns>'
                                   )
        dict_syll_true = {'ho-': ['h-', 'o-'], 'la:': ['l-', 'a:'], 'status': ['s-', 't-', 'a-', 't-', 'u-', 's:']}

        self.assertEqual(dict_syll, dict_syll_true)


    def test_tokenize_corpus(self):

        text = 'hola ! \n y tu como estás ? \n bien y tu ? \n'
        to_ignore = '''¿?¡!'''

        path = './data/test_utils_tokenize_corpus.txt'
        with open(path, 'w') as f: f.write(text)

        dict_word, dict_syll, freq_word, freq_syll = tokenize_corpus(path,
                                                                     to_ignore=to_ignore)
        dict_word_true = {'bien':['bien:'],
                          'como':['co-', 'mo:'],
                          'estás':['es-', 'tás:'],
                          'hola':['ho-','la:'],
                          'tu':['tu:'],
                          'y':['y:']
                          }
        dict_syll_true = {'bien:':['b-', 'i-', 'e-', 'n:'],
                          'co-':['c-','o-'],
                          'es-':['e-','s-'],
                          'ho-':['h-','o-'],
                          'la:':['l-','a:'],
                          'mo:':['m-','o:'],
                          'tás:':['t-','á-','s:'],
                          'tu:':['t-','u:'],
                          'y:':['y:']
                          }
        freq_word_true = {'bien':1,
                          'como':1,
                          'estás':1,
                          'hola':1,
                          'tu':2,
                          'y':2
                          }
        freq_syll_true = {'bien:':1,
                          'co-':1,
                          'es-':1,
                          'ho-':1,
                          'la:':1,
                          'mo:':1,
                          'tás:':1,
                          'tu:':2,
                          'y:':2
                          }

        self.assertEqual(dict_word, dict_word_true)
        self.assertEqual(dict_syll, dict_syll_true)
        self.assertEqual(freq_word, freq_word_true)
        self.assertEqual(freq_syll, freq_syll_true)


    def test_get_most_frequent(self):

        freq_dict = {'hola':1, 'como':5, 'estás':6, 'bien':2, 'mal':10}
        quantity = 0.4
        to_ignore = []
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'estás', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


        quantity = 0.8
        to_ignore = []
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'bien', 'como', 'estás', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


        quantity = 2
        to_ignore = []
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'estás', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


        quantity = 4
        to_ignore = []
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'bien', 'como', 'estás', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


        quantity = 5
        to_ignore = []
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'bien', 'como', 'estás', 'hola', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


        quantity = 1
        to_ignore = []
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'bien', 'como', 'estás', 'hola', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


        quantity = 5
        to_ignore = ['hola', 'bien']
        most_freq = get_most_frequent(freq_dict, quantity, to_ignore=to_ignore)
        most_freq_true = {'como', 'estás', 'mal'}

        self.assertEqual(most_freq, most_freq_true)


    def test_lprime(self):

        token_selected = ['ho-', 'la:', '<cm>', 'co-', 'mo:', 'es-', 'tás:', '<sp>', 'pa-', 'ra-', 'le-', 'le-', 'pi-',
                          'pe-', 'do:']
        sequence_length = 1
        lprime = Lprime(token_selected, sequence_length)

        self.assertEqual(lprime, 7)


        sequence_length = 2
        lprime = Lprime(token_selected, sequence_length)

        self.assertEqual(lprime, 8)


        token_selected = ['ho-', 'la:', '<cm>', 'co-', 'mo:', 'es-', 'tás:', '<sp>']
        sequence_length = 2
        lprime = Lprime(token_selected, sequence_length)

        self.assertEqual(lprime, 4)


    def test_ending_tokens_index(self):

        token_to_index = {'hola:':1, 'co-':2, 'mo:':3, 'es-':4, 'tas:':5, '<pt>':6}
        ends = [':', '>']
        ending, words_complete = ending_tokens_index(token_to_index, ends)
        ending_true = [1, 3, 5, 6]

        self.assertEqual(ending, ending_true)
        self.assertEqual(words_complete, 4)

    def test_get_syllables_to_ignore(self):

        words = ['hola', 'chao']
        dict_word_to_syll = {'chao':['cha-', 'o:'], 'hola':['ho-', 'la:']}

        syll_to_ignore = get_syllables_to_ignore(words, dict_word_to_syll)

        syll_true = ['cha-', 'ho-', 'la:', 'o:']

        for syll in syll_to_ignore:
            self.assertTrue(syll in syll_true)



if __name__ == '__main__':

    sys.path.append("..")

    from src.separadorSilabas import silabas

    from src.utils import preprocessing_file
    from src.utils import get_syllables
    from src.utils import get_characters
    from src.utils import get_freq_words
    from src.utils import get_freq_syllables
    from src.utils import word_to_syll
    from src.utils import syll_to_charac
    from src.utils import tokenize_corpus
    from src.utils import get_most_frequent
    from src.utils import Lprime
    from src.utils import ending_tokens_index
    from src.utils import get_syllables_to_ignore

    unittest.main()