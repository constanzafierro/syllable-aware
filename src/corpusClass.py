from process_corpus import *


class Corpus:
  
    def __init__(self,
               path_to_file,
               train_size,
               final_char=':',
               inter_char='-',
               sign_to_ignore=[],
               word_to_ignore=[]):
        
        self.path_to_file = path_to_file
        self.train_size = train_size
        self.final_char = final_char
        self.inter_char = inter_char
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

        self.tokenSelector = TokenSelector(final_char = self.final_char,
                           inter_char = self.inter_char,
                           sign_to_ignore = self.sign_to_ignore,
                           word_to_ignore= self.word_to_ignore)

        self.tokenSelector.get_dictionary(self.path_to_file)


    def select_tokens(self, quantity_word, quantity_syllables):
        
        self.tokenSelector.get_frequent(quantity_word = quantity_word,
                                        quantity_syll = quantity_syllable)

        self.token_selected = []
        with open(self.path_to_file) as f1:
                for line in f1:
                    words = line.lower().split()
                    for token in words:
                        token = token.strip()
                        tokenSelector.select(token, self.token_selected)


    def calculateLprime(self, sequent_length):
        self.lprime = Lprime(token_selected, sequent_length)
    
    def dictionaries_token_index(self):
        self.vocabulary = set(self.token_selected)
        self.token_to_index = dict((t, i) for i, t in enumerate(self.vocabulary, 1))
        self.index_to_token = dict((self.token_to_index[t], t) for t in self.vocabulary)
        self.ind_corpus = [self.token_to_index[token] for token in self.tokens] # corpus as indexes
        self.vocabulary_as_index = set(self.ind_corpus) # vocabualry as index
 
        len_train = int(len(self.ind_corpus)*self.train_size)
        self.train_set = self.ind_corpus[0:len_train] # indexes
        self.test_set = self.ind_corpus[len_train:] # indexes
        self.vocabulary_train = set(self.train_set) # indexes
        self.vocabulary_test = set(self.test_set) # indexes