from .TokenSelector import TokenSelector
from .utils import Lprime, ending_tokens_index
from .Generators import GeneralGenerator

import random # TODO: We must set a seed !!


class Corpus:
  
    def __init__(self,
                 path_to_file,
                 final_char,
                 final_punc,
                 inter_char,
                 signs_to_ignore,
                 words_to_ignore,
                 map_punctuation,
                 letters,
                 sign_not_syllable
                 ):

        '''

        :param path_to_file:
        :param final_char: ':'
        :param final_punc: '>'
        :param inter_char: '-'
        :param signs_to_ignore: []
        :param words_to_ignore: []
        :param map_punctuation: {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>',
                                  ',': '<cm>', '<unk>':'<unk>', ':':'<dc>', ';':'<sc>'}
        :param letters: 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        :param sign_not_syllable: '<sns>'

        '''

        self.path_to_file = path_to_file
        self.final_char = final_char
        self.final_punc = final_punc
        self.inter_char = inter_char
        self.signs_to_ignore = signs_to_ignore
        self.words_to_ignore = words_to_ignore

        self.map_punctuation = map_punctuation
        self.letters = letters

        self.sign_not_syllable = sign_not_syllable

        self.tokenSelector = TokenSelector(final_char = self.final_char,
                                           inter_char = self.inter_char,
                                           signs_to_ignore = self.signs_to_ignore,
                                           words_to_ignore = self.words_to_ignore,
                                           map_punctuation = self.map_punctuation,
                                           letters = self.letters,
                                           sign_not_syllable = self.sign_not_syllable
                                           )

        self.tokenSelector.get_dictionary(path_file = self.path_to_file)

        self.tokensplit = '<nl>' # for get_generators

        self.train_set = []
        self.eval_set = []

        self.quantity_word = None
        self.quantity_syllable = None

        self.token_selected = []

        self.lprime = 0

        self.vocabulary = set()
        self.token_to_index = dict()
        self.index_ends = []
        self.index_to_token = dict()
        self.ind_corpus = []
        self.vocabulary_as_index = set()

        self.vocabulary_train = set()
        self.vocabulary_eval = set()

        self.average_tpw = 1


    def set_tokens_selector(self, quantity_word, quantity_syllable):
        self.tokenSelector.get_frequent(quantity_word = quantity_word,
                                        quantity_syll = quantity_syllable
                                        )


    def select_tokens_from_file(self, path_to_file):
        token_selected = []

        with open(path_to_file) as f1:

                for line in f1:
                    words = line.lower().split()

                    words += ['\n']

                    for token in words:
                        token_selected = self.tokenSelector.select(token = token,
                                                                   tokens_selected = token_selected
                                                                   )

        return token_selected


    def set_token_selected(self):
        self.token_selected = self.select_tokens_from_file(self.path_to_file)


    def build_dictionaries(self):

        self.vocabulary = set(self.token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))

        self.index_ends, words_complete = ending_tokens_index(token_to_index=self.token_to_index,
                                                              ends=[self.final_char, self.final_punc]
                                                              )

        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in self.token_selected]  # corpus as indexes
        self.vocabulary_as_index = set(self.ind_corpus)  # vocabulary as index

        self.average_tpw = words_complete / len(self.ind_corpus)


    def set_lprime(self, sequence_length):
        self.lprime = Lprime(token_selected = self.token_selected,
                        sequence_length = sequence_length
                        )


    def get_parameters(self):
        return self.vocabulary, self.token_to_index, self.index_ends, self.index_to_token, self.average_tpw, self.lprime


    def split_corpus(self, percentage = 0, random = False, token_split= '<nl>', min_len = 0):

        if 0 <= percentage <= 100:
            percentage = percentage if percentage < 1 else percentage / 100.0
        else:
            raise (ValueError, "percentage = {} must be between zero and one hundred".format(percentage))

        val_set = []
        train_set = []

        if random:

            tokensplit = self.token_to_index[token_split]

            tokens = []

            for token in self.ind_corpus:

                tokens.append(token)

                if token == tokensplit:

                    if len(tokens) < min_len:
                        tokens = []
                        continue

                    p = random.choice(range(0, 100))

                    if p < percentage:
                        val_set += tokens

                    else:
                        train_set += tokens

                    tokens = []

        else:
            len_train = int(len(self.ind_corpus) * percentage)
            train_set = self.ind_corpus[0:len_train]  # indexes
            val_set = self.ind_corpus[len_train:]  # indexes

        return train_set, val_set