from __future__ import print_function
import numpy as np
import random
import sys
import re
from collections import Counter
from separadorSilabas import silabas
import unittest


def get_syllables(word, middle, end):
    '''Uses separadorSilabas to form a list of syllables
    Raises TypeError:
         if word is not separable i.e. if it's not a spanish word
    Args:
        word: string with the word to be separate
        middle: char to be added at the end of the middle syllables
        end: char to be added at the end of the end syllable
    Returns:
        List with syllables
    '''
    word_syllables = silabas(word).split('-')
    word_syllables = [word + middle for word in word_syllables]
    word_syllables[-1] = word_syllables[-1][0:-1] + end
    return word_syllables


def get_most_frequent(list_tokens, quantity):
    '''Selects most frequent tokens.
    Args:
        list_tokens: list of tokens from where to select
        quantity: number that indicates the elements to be select from the list, if it's between [0,1] it's used as percentage
    Returns:
        set of length less than or equal to quantity (percentage*len(different tokens)) containing the most frequent tokens
    '''
    count = 0
    ans = set()
    freq_dict = Counter(list_tokens).most_common()
    max_tokens = int(len(freq_dict)*quantity) if quantity <= 1 else quantity
    for token, freq in freq_dict:
        if len(ans) < max_tokens and token != '':
            ans.add(token)
    return ans


def get_next_word(string, begining_at):
    j = string[begining_at:].find(' ')
    return string[begining_at:begining_at+j] if j != -1 else string[begining_at:]


def token_to_string(tokens):
    map_punctuation = {'<ai>':'¿', '<ci>' : '?', '<pt>' : '.', '<nl>' : '\n', '<cm>' : ','}
    punctuation_sign = set(map_punctuation)
    string = ''
    for i, token in enumerate(tokens):
        if token in punctuation_sign:
            tokens[i] = map_punctuation[token]
        else:
            token[i] = token.replace('-', '').replace(':', ' ')
    return ''.join(tokens)


def get_list_words(corpus, sign_to_ignore, word_to_ignore):
    '''Returns the list of words in the corpus.'''
    no_newline = corpus.split('\n')
    new_corpus = []
    for s in no_newline:
        for sign in sign_to_ignore:
            s = s.replace(sign, '')
        new_corpus += s.split(' ')

    for i, word in enumerate(new_corpus):
        if word in word_to_ignore:
            new_corpus = list(filter(lambda a: a != word, new_corpus))
    return new_corpus


class TokenSelector():
    def __init__(self, final_char=':', inter_char='-'):
        self.final_char = final_char
        self.inter_char = inter_char

    def calculate_most_frequent(corpus, quantity):
        '''Selects and save as self most frequent tokens. If quantity 1 selects all of them.
        Args:
            corpus: string from where to select the tokens
            quantity: number to indicate how many tokens to select, if it's between 0-1 is considered as percentage.
        Returns:
            Nothing'''
        pass

    def select(corpus, i, tokens_selected):
        '''Appends in tokens_selected the next token starting at index i and returns updated i'''
        pass

# Implementations
class PuntuactionSelector(TokenSelector):
    def __init__(self, final_char=':', inter_char='-'):
        super().__init__(final_char, inter_char)
        self.map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>'}
        self.frequent = set(self.map_punctuation)

    def select(self, corpus, i, tokens_selected):
        if corpus[i] in self.frequent:
            tokens_selected.append(self.map_punctuation[corpus[i]])
            i += 1
        return i


class WordSelector(TokenSelector):
    def __init__(self, final_char=':', inter_char='-', sign_to_ignore=[], word_to_ignore=[]):
        '''
        Args:
            to_ignore: String containing characters to be ignored in the corpus'''
        super().__init__(final_char, inter_char)
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

    def calculate_most_frequent(self, corpus, quantity):
        corpus = get_list_words(corpus, self.sign_to_ignore, self.word_to_ignore)
        # asume que hay un ' ' que separa las palabras incluso al ignorar esos caracteres
        self.frequent = get_most_frequent(corpus, quantity)

    def select(self, corpus, i, tokens_selected):
        total_deleted = sum([1 for k, c in enumerate(corpus) if c in set(self.sign_to_ignore+ self.word_to_ignore) and k < i])
        corpus = ''.join([c for c in corpus if c not in set(self.sign_to_ignore+ self.word_to_ignore)])
        word = get_next_word(corpus, i - total_deleted)
        if word in self.frequent:
            tokens_selected.append(word+self.final_char)
            i += len(word)
        return i


class SyllableSelector(TokenSelector):
    def __init__(self, final_char=':', inter_char='-', sign_to_ignore=[], word_to_ignore=[]):
        super().__init__(final_char, inter_char)
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

    def calculate_most_frequent(self, corpus, quantity):
        new_corpus = get_list_words(corpus, self.sign_to_ignore, self.word_to_ignore)
        list_syll = []
        for word in new_corpus:
            try:
                list_syll += get_syllables(word, self.inter_char, self.final_char)
            except TypeError:
                print("word not considered for syllables: '{}'".format(word))
        self.frequent = get_most_frequent(list_syll, quantity)

    def select(self, corpus, i, tokens_selected):
        total_deleted = sum([1 for k,c in enumerate(corpus) if c in set(self.sign_to_ignore+self.word_to_ignore) and k<i])
        corpus = ''.join([c for c in corpus if c not in set(self.sign_to_ignore+ self.word_to_ignore)])
        word = get_next_word(corpus, i-total_deleted)
        try:
            syllables = get_syllables(word, self.inter_char, self.final_char)
            to_add = []
            len_corpus_added = 0
            for syll in syllables:
                if syll in self.frequent:
                    len_corpus_added += len(syll)-1
                    to_add.append(syll)
                else:
                    break
            tokens_selected += to_add
            return i + len_corpus_added
        except:
            return i


class CharacterSelector(TokenSelector): # letter
    def __init__(self, final_char=':', inter_char='-'):
        super().__init__()
        letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        letters += letters.upper()
        self.accepted_chars = set(letters)
        self.frequent = self.accepted_chars

    def select(self, corpus, i, tokens_selected):
        if corpus[i] in self.accepted_chars:
            if i+1<len(corpus) and corpus[i+1] in self.accepted_chars:
                tokens_selected.append(corpus[i] + self.inter_char)
            else:
                tokens_selected.append(corpus[i] + self.final_char)
            i += 1
        return i
