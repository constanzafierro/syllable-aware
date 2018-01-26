from token_selectors import *

def get_processed_text(raw_filename):
    '''Process raw text to tokens with the selectors array
    Args:
        raw_filename: string path to txt file
    Returns:
        Array with string tokens
    '''
    corpus = open(raw_filename).read().lower()
    not_word = ".,\n¡!:();\"0123456789…\xa0"
    word_selector = WordSelector(to_ignore=not_word)
    word_selector.calculate_most_frequent(corpus=corpus, quantity=0.6)
    syllable_selector = SyllableSelector(to_ignore=not_word)
    syllable_selector.calculate_most_frequent(corpus=corpus, quantity=0.4)
    selectors = [PuntuactionSelector(), word_selector, syllable_selector, CharacterSelector()]
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
