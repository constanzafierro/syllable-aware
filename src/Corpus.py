from .TokenSelector import TokenSelector
from .utils import Lprime, ending_tokens_index
from .Generators import GeneralGenerator


import random # TODO: We must set a seed !!


class Corpus:
  
    def __init__(self,
                 path_to_file,
                 train_size,
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
        :param train_size:
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
        self.train_size = train_size
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

        self.tokensplit = -1 # for get_generators

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

        self.train_ATPW = 1
        self.eval_ATPW = 1

    def select_tokens(self, quantity_word, quantity_syllable):
        
        self.quantity_word = quantity_word
        self.quantity_syllable = quantity_syllable

        self.tokenSelector.get_frequent(quantity_word = self.quantity_word,
                                        quantity_syll = self.quantity_syllable
                                        )

        token_selected = []

        with open(self.path_to_file) as f1:

                for line in f1:
                    words = line.lower().split()

                    for token in words:
                        token = token.strip()

                        token_selected = self.tokenSelector.select(token = token,
                                                                   tokens_selected = token_selected
                                                                   )

        self.token_selected = token_selected


    def calculateLprime(self, sequence_length):

        self.lprime = Lprime(token_selected = self.token_selected,
                             sequence_length = sequence_length
                             )
    
    
    def dictionaries_token_index(self):
        
        self.vocabulary = set(self.token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))

        self.index_ends = ending_tokens_index(token_to_index = self.token_to_index,
                                               ends = [self.final_char, self.final_punc]
                                               )

        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in self.token_selected] # corpus as indexes
        self.vocabulary_as_index = set(self.ind_corpus) # vocabualry as index
 
        len_train = int(len(self.ind_corpus)*self.train_size)

        self.train_set = self.ind_corpus[0:len_train] # indexes
        self.eval_set = self.ind_corpus[len_train:] # indexes

        self.vocabulary_train = set(self.train_set) # indexes
        self.vocabulary_eval = set(self.eval_set) # indexes


    def split_train_eval(self, val_percentage, token_split, min_len=0):

        self.vocabulary = set(self.token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))

        self.index_ends = ending_tokens_index(token_to_index = self.token_to_index,
                                               ends = [self.final_char, self.final_punc]
                                               )

        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in self.token_selected] # corpus as indexes
        self.vocabulary_as_index = set(self.ind_corpus) # vocabulary as index

        self.tokensplit = self.token_to_index[token_split]

        words_train_set = 0
        words_eval_set = 0
        qw = 0
        tokens = []

        for token in self.ind_corpus:

            tokens.append(token)

            if token in self.index_ends:
                qw += 1

            if token == self.tokensplit:

                if len(tokens) < min_len:
                    tokens = []
                    continue

                p = random.choice(range(0, 100))

                if p < val_percentage:
                    self.eval_set += tokens
                    words_eval_set += qw + 1
                    qw = 0

                else:
                    self.train_set += tokens
                    words_train_set += qw + 1
                    qw = 0

                tokens = []

        self.vocabulary_train = set(self.train_set) # indexes
        self.vocabulary_eval = set(self.eval_set) # indexes

        self.train_ATPW = words_train_set / len(self.train_set)
        self.eval_ATPW = words_eval_set / len(self.eval_set)


    def get_generators(self, batch_size):

        train_generator = GeneralGenerator(batch_size = batch_size,
                                           ind_tokens = self.train_set,
                                           vocabulary = self.vocabulary,
                                           max_len = self.lprime,
                                           split_symbol_index = self.tokensplit,
                                           count_to_split = -1
                                           )

        eval_generator = GeneralGenerator(batch_size = batch_size,
                                          ind_tokens = self.eval_set,
                                          vocabulary = self.vocabulary,
                                          max_len = self.lprime,
                                          split_symbol_index = self.tokensplit,
                                          count_to_split = -1
                                          )

        return train_generator, eval_generator
