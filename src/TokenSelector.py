from utils import tokenize_corpus, get_most_frequent


# TODO: Testear métodos y clases ...
# TODO: Pasar Punctuation y Characters como inputs de TokenSelector
# TODO: Implementar dichas modificaciones a los inputs en la clase Corpus
# TODO: Agregar dichas modificaciones a lstm.py y posteriormente a main.py


class TokenSelector():

    def __init__(self,
                 final_char=':',
                 inter_char='-',
                 sign_to_ignore=[],
                 word_to_ignore=[],
                 map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>', '<unk>':'<unk>', ':':'<dc>', ';':'<sc>'}
                 letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
                 ):

        self.final_char = final_char
        self.inter_char = inter_char
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

        # Punctuation
        self.map_punctuation = map_punctuation
        self.punctuation = set(self.map_punctuation)

        # Characters
        self.characters = set(letters)


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
                            if c in self.characters:
                                tokens_selected.append(c)

        return tokens_selected
