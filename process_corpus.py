import re
import operator
from separadorSilabas import silabas


def get_freq_words(word, freq_word, to_ignore):
    if word in to_ignore:
        return freq_word

    if word in freq_word:
        freq_word[word] += 1
    else:
        freq_word[word] = 1

    return freq_word


def get_freq_syllables(freq_word, dict_word, to_ignore):
    freq_syll = dict()
    for k,v in freq_word.items():
        if k in to_ignore:
            pass

        try:
            syllables = dict_word[k]
        except KeyError:
            continue

        for s in syllables:
            if s in freq_syll:
                freq_syll[s] += v
            else:
                freq_syll[s] = v
    return freq_syll


def add_space_file(path_in, path_out):
    with open(path_out, 'w') as f1:
        with open(path_in) as f2:
            for line in f2:
                s = re.sub('([.,;:¿¡!?"()\´])', r' \1 ', line)
                s = re.sub('\s{2,}', ' ', s)
                f1.write(s+'\n')


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


def get_characters(syllable, middle = '-', end = ':'):
    letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
    s = []
    for letter in syllable:
        if letter in letters:
            s += [letter + middle]

    if syllable[-1] == end:
        s[-1] = s[-1].replace(middle, end)

    return s


def word_to_syll(word, dict_word, to_ignore = [], middle='-', end=':'):
    if word in to_ignore:
        return dict_word

    if word not in dict_word:
        try:
            dict_word[word] = get_syllables(word, middle, end)
        except TypeError:
            print("word not considered for syllables: '{}'".format(word))
    return dict_word


def syll_to_charac(word, dict_syll, dict_word, to_ignore = [], middle='-', end=':'):
    if word in to_ignore:
        return dict_syll
    try:
        syllables = dict_word[word]
    except KeyError:
        print("word not exist in dictionary: '{}'".format(word))
        return dict_syll

    for syll in syllables:
        if syll not in dict_syll:
            dict_syll[syll] = get_characters(syll, middle, end)

    return dict_syll


def tokenize_corpus(path_file, to_ignore = []):
    dict_word = dict()
    dict_syll = dict()
    freq_word = dict()

    with open(path_file, 'r') as f1:
        for line in f1:
            words = line.lower().split()
            for w in words:
                dict_word = word_to_syll(w, dict_word, to_ignore)
                dict_syll = syll_to_charac(w, dict_syll, dict_word, to_ignore)
                freq_word = get_freq_words(w, freq_word, to_ignore)
    freq_syll = get_freq_syllables(freq_word, dict_word, to_ignore)

    return dict_word, dict_syll, freq_word, freq_syll


def get_most_frequent(freq_dict, quantity, to_ignore = []):
    '''Selects most frequent tokens.
    Args:
        freq_dict: list of tokens and frequent in corpus from where to select
        quantity: number that indicates the elements to be select from the list, if it's between [0,1] it's used as percentage
    Returns:
        set of length less than or equal to quantity (percentage*len(different tokens)) containing the most frequent tokens
    '''
    most_freq = set()
    max_tokens = int(len(freq_dict)*quantity) if quantity <= 1 else quantity
    #order = sorted(freq_dict.items(), key=freq_dict.get, reverse=True)
    for token, freq in sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True):
        if len(most_freq) < max_tokens:
           if token not in to_ignore:
                most_freq.add(token)
    return most_freq


############################### CLASS TOKEN SELECTOR ##########################

class TokenSelector():

    def __init__(self, final_char=':', inter_char='-', sign_to_ignore=[], word_to_ignore=[]):
        self.final_char = final_char
        self.inter_char = inter_char
        self.sign_to_ignore = sign_to_ignore
        self.word_to_ignore = word_to_ignore

        # Punctuation
        self.map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>', '"':'<qt>'}
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

############################### END CLASS TOKEN SELECTOR ######################


def main(flag):
    path_in = 'reclamos_cl.txt'
    path_out = 'reclamos_cl_add_space.txt'

    if flag:
        add_space_file(path_in, path_out)

    tokenSelector = TokenSelector()
    tokenSelector.get_dictionary(path_out)
    tokenSelector.get_frequent(quantity_word = 1.0, quantity_syll = 0.0)
    #print(tokenSelector.syllables)

    token_selected = []
    with open(path_out) as f1:
        for line in f1:
            words = line.split()
            for token in words:
                tokenSelector.select(token, token_selected)

    sequence_length = len(token_selected)
    print(Lprime(token_selected,sequence_length), len(token_selected))



def Lprime(token_selected, sequence_length):
    tokens_len = []
    count_tokens = 0
    maxL = 0

    for token in token_selected:
        count_tokens += 1
        if token[-1] == ':' or token[-1] == '>':
            tokens_len.append(count_tokens)
            count_tokens = 0
        if len(tokens_len) == sequence_length:
            Lpocket = sum(tokens_len)
            tokens_len = tokens_len[1:]
            if Lpocket > maxL:
                maxL = Lpocket

    return maxL

if __name__ == '__main__':
    main(False)