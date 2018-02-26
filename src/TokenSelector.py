from .utils import tokenize_corpus, get_most_frequent, get_syllables_to_ignore


class TokenSelector():

    def __init__(self,
                 final_char,
                 inter_char,
                 signs_to_ignore,
                 words_to_ignore,
                 map_punctuation,
                 letters,
                 sign_not_syllable
                 ):

        self.final_char = final_char
        self.inter_char = inter_char
        self.signs_to_ignore = signs_to_ignore
        self.words_to_ignore = words_to_ignore

        # Punctuation
        self.map_punctuation = map_punctuation
        self.punctuation = set(self.map_punctuation)

        # Characters
        self.letters = set(letters)

        self.sign_not_syllable = sign_not_syllable

        #dictionaries
        self.dict_word = dict()
        self.dict_syll = dict()
        self.freq_word = dict()
        self.freq_syll = dict()

        #selectors
        self.words = set()
        self.syllables = set()

    def get_dictionary(self, path_file):

        to_ignore = [i for i in self.punctuation]
        to_ignore = to_ignore + self.signs_to_ignore + self.words_to_ignore

        self.dict_word, self.dict_syll, self.freq_word, self.freq_syll = tokenize_corpus(path_file = path_file,
                                                                                         to_ignore = to_ignore,
                                                                                         middle = self.inter_char,
                                                                                         end = self.final_char,
                                                                                         sign_not_syllable = self.sign_not_syllable
                                                                                         )


    def get_frequent(self, quantity_word, quantity_syll):

        self.words = get_most_frequent(freq_dict = self.freq_word,
                                       quantity = quantity_word,
                                       to_ignore = []
                                       )

        # count syllables ignoring most frequent words

        syll_to_ignore = get_syllables_to_ignore(self.words, self.dict_word)

        self.syllables = get_most_frequent(freq_dict =  self.freq_syll,
                                           quantity = quantity_syll,
                                           to_ignore = syll_to_ignore
                                           )

    def select(self, token, tokens_selected):

        if token in self.punctuation:
            tokens_selected.append(self.map_punctuation[token])

        elif token in self.dict_word:

            if token in self.words:
                tokens_selected.append(token + self.final_char)

            else:
                for s in self.dict_word[token]:

                    if s in self.syllables:
                        tokens_selected.append(s)

                    else:
                        if s == self.sign_not_syllable:
                            for c in self.dict_syll[token]:
                                if c[0] in self.letters:

                                    tokens_selected.append(c)
                        else:
                            for c in self.dict_syll[s]:
                                if c[0] in self.letters:
                                    tokens_selected.append(c)

        return tokens_selected
