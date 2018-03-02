from .TokenSelector import TokenSelector
from .utils import Lprime, ending_tokens_index

import random  # TODO: We must set a seed !!
import os
import json


class Tokenization:

    def __init__(self,
                 path_to_file,
                 final_char,
                 final_punc,
                 inter_char,
                 signs_to_ignore,
                 words_to_ignore,
                 map_punctuation,
                 letters,
                 sign_not_syllable,
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

        self.tokenSelector = TokenSelector(final_char=self.final_char,
                                           inter_char=self.inter_char,
                                           signs_to_ignore=self.signs_to_ignore,
                                           words_to_ignore=self.words_to_ignore,
                                           map_punctuation=self.map_punctuation,
                                           letters=self.letters,
                                           sign_not_syllable=self.sign_not_syllable
                                           )

        if self.path_to_file != None:
            self.tokenSelector.setting_dictionaries(path_file=self.path_to_file)

        ## parameters tokenization
        self.lprime = 0
        self.quantity_word = 0
        self.quantity_syll = 0
        self.tokensplit = ''
        self.vocabulary = set()
        self.token_to_index = dict()
        self.index_ends = []
        self.index_to_token = dict()
        self.ind_corpus = []
        self.average_tpw = 1

    def set_tokenSelector_params(self, params):
        self.tokenSelector.set_params(params)

    def setting_tokenSelector_params(self, quantity_word, quantity_syllable):
        self.quantity_word = quantity_word
        self.quantity_syll = quantity_syllable
        self.tokenSelector.setting_selectors(quantity_word=self.quantity_word,
                                             quantity_syll=self.quantity_syll
                                             )

    def save_tokenSelector(self, path_to_file):

        params = self.tokenSelector.params()

        if os.path.exists(path=path_to_file):
            Warning(FileExistsError, "Warning path exists, '{}'".format(path_to_file))

        with open(path_to_file, "w") as f:
            json.dump(params, f)

    def load_tokenSelector(self, path_to_file):

        if not os.path.exists(path=path_to_file):
            raise FileNotFoundError("Path doesn't exists, '{}'".format(path_to_file))

        with open(path_to_file) as f:
            params = json.load(f)

        self.tokenSelector.set_params(params)
        return params

    def select_tokens(self, path_to_file=None):

        path_to_file = path_to_file if path_to_file != None else self.path_to_file

        if not os.path.exists(path=path_to_file):
            raise FileNotFoundError("Path not exists, '{}'".format(path_to_file))

        token_selected = []

        with open(path_to_file) as f1:

            for line in f1:
                words = line.lower().split()

                words += ['\n']

                for token in words:
                    token_selected = self.tokenSelector.select(token=token,
                                                               tokens_selected=token_selected
                                                               )

        return token_selected

    def setting_experiment(self, token_selected, sequence_length):

        self.vocabulary = set(token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))

        self.index_ends, words_complete = ending_tokens_index(token_to_index=self.token_to_index,
                                                              ends=[self.final_char, self.final_punc]
                                                              )

        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in token_selected]  # corpus as indexes

        self.average_tpw = compute_tpw(self.index_to_token, self.index_ends)

        self.lprime = Lprime(token_selected=token_selected,
                             sequence_length=sequence_length
                             )

    def set_params_experiment(self, params):

        params_key = ["tokensplit", "quantity_word", "quantity_syll", "lprime",
                      "vocabulary", "token_to_index", "index_to_token", "index_ends",
                      "average_tpw"]
        for param, value in params:
            if param not in params_key:
                raise KeyError("params doesn't contain {}".format(param))

        self.tokensplit = params["tokensplit"]
        self.quantity_word = params["quantity_word"]
        self.quantity_syll = params["quantity_syll"]
        self.lprime = params["lprime"]
        self.vocabulary = params["vocabulary"]
        self.token_to_index = params["token_to_index"]
        self.index_to_token = params["index_to_token"]
        self.index_ends = params["index_ends"]
        self.average_tpw = params["average_tpw"]

    def params_experiment(self):

        params = {"tokensplit": self.tokensplit,
                  "quantity_word": self.quantity_word,
                  "quantity_syll": self.quantity_syll,
                  "lprime": self.lprime,
                  "vocabulary": list(self.vocabulary),
                  "token_to_index": self.token_to_index,
                  "index_to_token": self.index_to_token,
                  "index_ends": self.index_ends,
                  "average_tpw": self.average_tpw,
                  }
        return params

    def save_experiment(self, path_to_file):

        params = self.params_experiment()

        if os.path.exists(path=path_to_file):
            Warning("Warning path exists, '{}'".format(path_to_file))

        with open(path_to_file, "w") as f:
            json.dump(params, f)

    def load_experiment(self, path_to_file):
        if not os.path.exists(path=path_to_file):
            raise FileNotFoundError("Path doesn't exists, '{}'".format(path_to_file))

        with open(path_to_file) as f:
            params = json.load(f)

        self.set_params_experiment(params)
        return params

    def split_train_val(self, train_size, token_split, random_split=False, min_len=0):

        if 0 < train_size < 100:
            train_size = train_size if train_size < 1 else train_size / 100
        else:
            raise ValueError("train_size = '{}' must be between zero and one hundred".format(train_size))

        if token_split not in self.token_to_index:
            raise KeyError("token_split '{}' isn't in vocabulary".format(token_split))

        val_set = []
        train_set = []

        if random_split:

            tokensplit = self.token_to_index[token_split]

            tokens = []

            for token in self.ind_corpus:

                tokens.append(token)

                if token == tokensplit:

                    if len(tokens) < min_len:
                        tokens = []
                        continue

                    if len(train_set) == 0:
                        train_set += tokens
                        continue
                    elif len(val_set) == 0:
                        val_set += tokens
                        continue

                    p = random.choice(range(0, 100))

                    if p > train_size*100:
                        val_set += tokens

                    else:
                        train_set += tokens

                    tokens = []

        else:
            len_train = int(len(self.ind_corpus) * train_size)
            train_set = self.ind_corpus[0:len_train]  # indexes
            val_set = self.ind_corpus[len_train:]  # indexes

        return train_set, val_set

    def converting_token_to_index(self, token_selected):
        index_array = []
        for token in token_selected:
            if token in self.token_to_index:
                index_array.append(self.token_to_index[token])

        return index_array

    def converting_index_to_token(self, index_selected):
        token_array = []
        for index in index_selected:
            if index in self.index_to_token:
                token_array.append(self.index_to_token[index])

        return token_array

    def coverage(self, path_to_file):
        """
        :param path_to_file:
        :return: percentage_words_with_words, percentage_words_with_syllables, percentage_syllables_with_syllables
        """

        count = [0, 0, 0, 0]
        with open(path_to_file) as f1:

            for line in f1:
                words = line.lower().split()

                words += ['\n']

                for token in words:
                    count = self.tokenSelector.coverage(token=token, count=count)

        return 100.0 * count[1] / count[0], 100.0 * count[2] / count[0], 100.0 * count[3] / count[0]


def compute_tpw(index_to_token, index_ends):
    words_complete = 0
    len_tokens = len(index_to_token)
    if len_tokens == 0:
        raise ValueError("Dictionary index_to_token can't be empty")

    for index in index_to_token:
        if index in index_ends:
            words_complete += 1

    return len_tokens / words_complete