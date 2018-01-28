from token_selectors import *


def helper_get_processed_text(raw_filename, quantity_word = 0.5, quantity_syllable = 0.5):
    '''Process array of string to tokens with the selectors array
    Args:
        raw_filename: path of file in txt
        quantity_word: quantity of words if quantity of words is between 0 and 1 is a percent
                       by default quantity_word = 0.5
        quantity_syllable: quantity of syllable if quantity of words is between 0 and 1 is a percent
                       by default quantity_syllable = 0.5
    Returns:
        Array with string tokens
    '''
    corpus = open(raw_filename).read().lower()
    selectors = get_selectors(corpus, quantity_word, quantity_syllable)
    return get_processed_text(corpus, selectors)


def get_selectors(corpus, quantity_word = 0.5, quantity_syllable = 0.5):
    '''Generate a selectors from array of string and the quantity of words and syllables
    Args:
        corpus: strings
        quantity_word: quantity of words if quantity of words is between 0 and 1 is a percent
                       by default quantity_word = 0.5
        quantity_syllable: quantity of syllable if quantity of words is between 0 and 1 is a percent
                       by default quantity_syllable = 0.5
    Returns:
        Array with tokens selectors
    '''
    #corpus = open(raw_filename).read().lower()#[1:1000]
    not_word = ".,\n¡!:();\"0123456789…\xa0"

    sign_to_ignore = [i for i in not_word]

    word_selector = WordSelector(sign_to_ignore=sign_to_ignore)
    word_selector.calculate_most_frequent(corpus=corpus, quantity=quantity_word)

    word_to_ignore = [i for i in word_selector.frequent]

    syllable_selector = SyllableSelector(sign_to_ignore=sign_to_ignore, word_to_ignore = word_to_ignore)
    syllable_selector.calculate_most_frequent(corpus=corpus, quantity=quantity_syllable)

    #print(syllable_selector.frequent)
    ## Para toquenizar se tokeniza primero en puntuacion, luego en palabras

    selectors = [PuntuactionSelector(), word_selector, syllable_selector, CharacterSelector()]

    return selectors


def get_processed_text(corpus, selectors):
    '''Process array of string to tokens with the selectors array
    Args:
        corpus: strings
        selectors: array of selectos
    Returns:
        Array with string tokens
    '''
    #corpus = open(raw_filename).read().lower()#[1:1000]
    processed_corpus = []
    i = 0
    while i < len(corpus)-6:
        j = i
        for token_selector in selectors:
            while j < len(corpus)-6:
                k = token_selector.select(corpus, j, processed_corpus)
                if k==j:
                    break
                else:
                    j = k
        if i == j:
            # none selector picked it -> ignore it
            i += 1
        else:
            i = j
    return processed_corpus
