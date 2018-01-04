from language_model_general import *

corpus = open('horoscopo_raw.txt').read().lower()[0:1000]
not_word = ".,\n¡!:();\"0123456789…\xa0"
word_selector = WordSelector(to_ignore=not_word)
word_selector.calculate_most_frequent(corpus=corpus, quantity=0.6) # 2
syllable_selector = SyllableSelector(to_ignore=not_word)
syllable_selector.calculate_most_frequent(corpus=corpus, quantity=0.4) # 3
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
        #processed_corpus.append(corpus[i])
        i += 1
    else:
        i = j
print('---------- RESULT ---------')
print(''.join(processed_corpus))
