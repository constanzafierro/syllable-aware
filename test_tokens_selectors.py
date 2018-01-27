import unittest
from token_selectors import *

def test_token(corpus, token):
    '''Iterates over the corpus calling select method of token object
    Returns:
        String with corpus changed by select
    '''
    processed_corpus = []
    i = 0
    while(i < len(corpus)):
        j = token.select(corpus, i, processed_corpus)
        if j == i:
            processed_corpus.append(corpus[i])
            i += 1
        else:
            i = j
    return

class TestStringMethods(unittest.TestCase):

    def test_getting_most_frequent(self):
        tokens = [1,2,2,3,3,3,4,4,4,4]
        self.assertEqual(list(get_most_frequent(tokens, 1)), [1,2,3,4])
        self.assertEqual(list(get_most_frequent(tokens, 0.8)), [2,3,4])
        self.assertEqual(list(get_most_frequent(tokens, 0.5)), [3,4])
        self.assertEqual(list(get_most_frequent(tokens, 0.3)), [4])
        #self.assertTrue('FOO'.isupper())
        #self.assertFalse('Foo'.isupper())

    def test_puntuaction(self):
        corpus = "Hola ¿como estas? Bien! Y tu? No tan bien..."
        puntuaction_selctor = PuntuactionSelector()
        processed_corpus = []
        i = 0
        while(i < len(corpus)):
            j = puntuaction_selctor.select(corpus, i, processed_corpus)
            if j == i:
                processed_corpus.append(corpus[i])
                i += 1
            else:
                i = j
        self.assertEqual(''.join(processed_corpus),
                         "Hola <ai>como estas<ci> Bien! Y tu<ci> No tan bien<pt><pt><pt>")

    def test_word(self):
        corpus = "Hola ¿como estas? bien! Y tu? No tan bien, como..."
        word_selector = WordSelector(sign_to_ignore=[i for i in "¿?.,\n¡!:();\"0123456789…"])
        word_selector.calculate_most_frequent(corpus=corpus, quantity=0.2)
        processed_corpus = []
        i = 0
        while(i < len(corpus)):
            j = word_selector.select(corpus, i, processed_corpus)
            i = i+1 if j==i else j
        # 8 different words -> 0,2 is just the most frequent words
        self.assertEqual(processed_corpus, ["como:", "como:"])

        word_selector.calculate_most_frequent(corpus=corpus, quantity=0.3)
        processed_corpus = []
        i = 0
        while(i < len(corpus)):
            j = word_selector.select(corpus, i, processed_corpus)
            i = i+1 if j==i else j
        self.assertEqual(processed_corpus, ["como:", "bien:", "bien:", "como:"])

    def test_syllable(self):
        corpus = "Hola corazón ¿como estas? bien! Y tu? No tan bien, como..."
        # ho-la: co-ra-zón: co-mo: es-tas: bien: y: tu: No: tan: bien: co-mo:
        # 13, co:3 mo:2 bien:2
        syllable_selector = SyllableSelector(sign_to_ignore=[i for i in "¿?.,\n¡!:();\"0123456789…"])
        # 1 silaba
        syllable_selector.calculate_most_frequent(corpus=corpus, quantity=0.1)
        processed_corpus = []
        i = 0
        while(i < len(corpus)):
            j = syllable_selector.select(corpus, i, processed_corpus)
            i = i+1 if j==i else j
        self.assertEqual(processed_corpus, ["co-", "co-", "co-"])

        # 3 silaba
        syllable_selector.calculate_most_frequent(corpus=corpus, quantity=0.3)
        processed_corpus = []
        i = 0
        while(i < len(corpus)):
            j = syllable_selector.select(corpus, i, processed_corpus)
            i = i+1 if j==i else j
        self.assertEqual(processed_corpus, ["co-", "co-", "mo:", "bien:", "bien:", "co-", "mo:"])

    def test_character(self):
        corpus = "Hola ¿como estas?"
        character_selector = CharacterSelector()
        processed_corpus = []
        i = 0
        while(i < len(corpus)):
            j = character_selector.select(corpus, i, processed_corpus)
            i = i+1 if j==i else j
        self.assertEqual(processed_corpus, ['H-','o-', 'l-', 'a:', 'c-', 'o-', 'm-', 'o:', 'e-', 's-', 't-', 'a-', 's:'])

    def test_combination(self):
        corpus = "Hola coco ¿Cómo estás? bien! Y tú? No tan bien, como..."
        ignore = [i for i in "¿?.,\n¡!:();\"0123456789…"]
        word_selector = WordSelector(sign_to_ignore=ignore)
        word_selector.calculate_most_frequent(corpus=corpus, quantity=0.2) # 2
        syllable_selector = SyllableSelector(sign_to_ignore=ignore)
        syllable_selector.calculate_most_frequent(corpus=corpus, quantity=0.25) # 3
        selectors = [PuntuactionSelector(), word_selector, syllable_selector, CharacterSelector()]
        processed_corpus = []
        i = 0
        while i < len(corpus):
            j = i
            for token_selector in selectors:
                while j < len(corpus):
                    k = token_selector.select(corpus, j, processed_corpus)
                    if k==j:
                        break
                    else:
                        j = k
            i = j if j != i else i+1

        self.assertEqual(''.join(processed_corpus),
                         "Hola:co-c-o:<ai>C-ó-m-o:e-s-t-á-s:<ci>bien:Y:t-ú:<ci>N-o:t-a-n:bien:<cm>co-mo:<pt><pt><pt>")

    def test_quantity(self):
        # Same as before but with quantity of words/syllables
        corpus = "Hola coco ¿Cómo estás? bien! Y tú? No tan bien, como..."
        not_word = [i for i in "¿?.,\n¡!:();\"0123456789…"]
        word_selector = WordSelector(sign_to_ignore=not_word)
        word_selector.calculate_most_frequent(corpus=corpus, quantity=2) # 2
        syllable_selector = SyllableSelector(sign_to_ignore=not_word)
        syllable_selector.calculate_most_frequent(corpus=corpus, quantity=3) # 3
        selectors = [PuntuactionSelector(), word_selector, syllable_selector, CharacterSelector()]
        processed_corpus = []
        i = 0
        while i < len(corpus):
            j = i
            for token_selector in selectors:
                while j < len(corpus):
                    k = token_selector.select(corpus, j, processed_corpus)
                    if k==j:
                        break
                    else:
                        j = k
            i = j if j != i else i+1

        self.assertEqual(''.join(processed_corpus),
                         "Hola:co-c-o:<ai>C-ó-m-o:e-s-t-á-s:<ci>bien:Y:t-ú:<ci>N-o:t-a-n:bien:<cm>co-mo:<pt><pt><pt>")

if __name__ == '__main__':
    unittest.main()
