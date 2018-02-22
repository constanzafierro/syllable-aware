from utils import tokenize_corpus, get_most_frequent


class TokenSelector():

    def __init__(self, final_char=':', inter_char='-', sign_to_ignore=[], word_to_ignore=[]):
        self.final_char = final_char
        self.inter_char = inter_char
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

        # Punctuation
        self.map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>', '<unk>':'<unk>', ':':'<dc>', ';':'<sc>'}
        self.punctuation = set(self.map_punctuation)

        # Character
        letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
        letters += letters.upper()
        self.accepted_chars = set(letters)
        self.characters = self.accepted_chars


    def get_dictionary(self, path_file):
        to_ignore = [i for i in self.punctuation]
        to_ignore = to_ignore + self.sign_to_ignore + self.word_to_ignore
        self.dict_word, self.dict_syll, self.freq_word, self.freq_syll = tokenize_corpus(path_file, to_ignore = to_ignore)


    def get_frequent(self, quantity_word, quantity_syll):
        self.words = get_most_frequent(self.freq_word, quantity_word)
        # count syllables ignoring most frequent words
        self.syllables = get_most_frequent(self.freq_syll, quantity_syll, [w for w in self.words])


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
                        for c in self.dict_syll[s]:
                            tokens_selected.append(c)
        return tokens_selected
