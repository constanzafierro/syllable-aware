import re
import operator
import os
from .separadorSilabas import silabas


# TODO: Agregar 'docstrings' a cada método explicando inputs y outputs o su comportamiento


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
            continue

        syllables = dict_word[k]
#        try:
#            syllables = dict_word[k]
#        except KeyError:
#            continue

        for s in syllables:

            if s in freq_syll:
                freq_syll[s] += v

            else:
                freq_syll[s] = v

    return freq_syll


def preprocessing_file(path_in, path_out, to_ignore, punctuation, token_unknow):

    if not os.path.exists(path_in):
        raise TypeError("File not exists {0}".format(path_in))

    with open(path_out, 'w') as f1:
        with open(path_in) as f2:

            for line in f2:
                words = line.lower().split()
                words_array = []
                for word in words:
                    if word in to_ignore:
                        continue
                    elif word in punctuation:
                        words_array.append(word)
                        continue
                    try:
                        syll = get_syllables(word=word, middle='-', end=':')
                        words_array.append(word)
                    except:
                        print("Word '{}' does not belong to Spanish, was replaced by '{}'".format(word, token_unknow))
                        words_array.append(token_unknow)

                s = " ".join(words_array)
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


def get_characters(token,
                   middle,
                   end,
                   letters):

    s = []

    for letter in token:
        if letter in letters:
            s += [letter + middle]

    if token[-1] == end:
        s[-1] = s[-1].replace(middle, end)

    return s


def word_to_syll(word, dict_word, to_ignore, middle, end, sign_not_syllable, verbose):

    if word in to_ignore:
        return dict_word

    if word not in dict_word:
        try:
            dict_word[word] = get_syllables(word = word,
                                            middle = middle,
                                            end = end
                                            )
        except TypeError:
            if verbose:
                print("Word not considered for function word_to_syll: '{}'".format(word))
            dict_word[word] = [sign_not_syllable]

    return dict_word


def syll_to_charac(word, dict_syll, dict_word, to_ignore, middle, end, sign_not_syllable, letters):

    if word in to_ignore:
        return dict_syll

#    try:
#        syllables = dict_word[word]
#    except KeyError:
#        print("word not exist in dictionary: '{}'".format(word))
#        return dict_syll

    syllables = dict_word[word]

    for syll in syllables:

        if syll not in dict_syll:

            if syll == sign_not_syllable:
                dict_syll[word] = get_characters(token = word + end,
                                                 middle = middle,
                                                 end = end,
                                                 letters = letters
                                                 )
            else:
                dict_syll[syll] = get_characters(token = syll,
                                                 middle = middle,
                                                 end = end,
                                                 letters = letters
                                                 )

    return dict_syll


def tokenize_corpus(path_file,
                    to_ignore,
                    middle = '-',
                    end = ':',
                    sign_not_syllable = '<sns>',
                    verbose = False,
                    letters = 'aáeéoóíúiuübcdfghjklmnñopqrstvwxyz'
                    ):

    dict_word = dict()
    dict_syll = dict()
    freq_word = dict()

    with open(path_file, 'r') as f1:

        for line in f1:
            words = line.lower().split()

            for word in words:
                word = word.strip()

                if word == '': continue

                dict_word = word_to_syll(word = word,
                                         dict_word = dict_word,
                                         to_ignore = to_ignore,
                                         middle = middle,
                                         end = end,
                                         sign_not_syllable = sign_not_syllable,
                                         verbose = verbose
                                         )

                dict_syll = syll_to_charac(word = word,
                                           dict_syll = dict_syll,
                                           dict_word = dict_word,
                                           to_ignore = to_ignore,
                                           middle = middle,
                                           end = end,
                                           sign_not_syllable = sign_not_syllable,
                                           letters = letters
                                           )

                freq_word = get_freq_words(word = word,
                                           freq_word = freq_word,
                                           to_ignore = to_ignore
                                           )

    freq_syll = get_freq_syllables(freq_word = freq_word,
                                   dict_word = dict_word,
                                   to_ignore = to_ignore
                                   )

    return dict_word, dict_syll, freq_word, freq_syll


def get_most_frequent(freq_dict, quantity, to_ignore):

    '''Selects most frequent tokens.
    Args:
        freq_dict: list of tokens and frequent in corpus from where to select
        quantity: number that indicates the elements to be select from the list, if it's between [0,1] it's used as percentage
        to_ignore: array of elements to ignore
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


def ending_tokens_index(token_to_index, ends):

    token_end = []
    words_complete = 0
    for k,v in token_to_index.items() :
        if k[-1] in ends:
            words_complete += 1
            token_end.append(v)

    return token_end, words_complete


def get_syllables_to_ignore(words, dict_word_to_syll, verbose = False):

    syll_to_ignore = []

    for w in words:
        try:
            syll_to_ignore += dict_word_to_syll[w]
        except KeyError:
            if verbose:
                print('KeyError in dict_word_to_syll, word = {}'.format(w))

    return syll_to_ignore

