import kmp as km
import numpy as np
from process_text import *


def next_word_generative(model, sentence, index_to_token):
    ## Generation of the next word
    last_char = ''
    word_generate = ''
    while last_char != ':':
        x_pred = np.zeros((1, len(sentence)))
        for t, token in enumerate(sentence):
            x_pred[0, t] = token_to_index[token]
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds) # temperature = 1.0 por defecto
        next_token = index_to_token[next_index+1] # dict starting at 1
        next = index_to_token[np.argmax(preds)+1]
        sentence = sentence[1:] + [next_token]
        word_generate += next_token
        last_char = next_token[-1]
    return word_generate


def test_eval(model, index_to_token, corpus, selectors, step_t = 100):

    only_word = get_selectors(corpus, quantity_word = 1.0, quantity_syllable = 0.0)
    Ntest = len(get_processed_text(corpus, only_word))

    token_test = get_processed_text(corpus, selectors)

    start_index = 1
    ppl = 0
    for i in range(len(corpus) - step_t - 1):
        #start_index
        if i < step_t:
            words = corpus[start_index: start_index + i]
            token_test = get_processed_text(words, selectors)
            sentence = tokens_test if len(tokens_test) < step_t else tokens_test[-step_t:]
            word_i = next_word_generative(model, sentence, index_to_token)
            word_i_processed = word_i.replace("-", "")
            word_i_processed = word_i_processed.replace(":", "")
            ppl += np.log(perplexity_i(word_i_processed, words, corpus))
        else:
            words = corpus[start_index: start_index + step_t]
            token_test = get_processed_text(words, selectors)
            sentence = tokens_test if len(tokens_test) < step_t else tokens_test[-step_t:]
            word_i = next_word_generative(model, sentence, index_to_token)
            word_i_processed = word_i.replace("-", "")
            word_i_processed = word_i_processed.replace(":", "")
            ppl += np.log(perplexity_i(word_i_processed, words, corpus))
            start_index += 1

    return np.exp(-ppl/Ntest)


def perplexity_i(word_i_processed, words, corpus):
    indexes = km.kmpMatch(words, corpus)
    p_word = 0
    p_contex = 0

    if len(indexes) == 0:
        return 0.01

    for i in indexes:
        p_contex += 1
        if corpus[i:(i+len(words)+1)] == (words + word_i_processed):
            p_word += 1

    return p_word/p_context