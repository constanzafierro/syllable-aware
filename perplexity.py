import kmp as km
import numpy as np
from process_text import *


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    text = string_tokens
    start_index = random.randint(0, len(text) - max_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        sentence = text[start_index: start_index + max_len]
        generated = sentence.copy()
        print('----- Generating with seed: "' + ''.join(sentence) + '"')
        #sys.stdout.write(''.join(generated))

        for i in range(100):
            x_pred = np.zeros((1, max_len)) # no debería ser en len(sentence) ?? 
            for t, token in enumerate(sentence):
                x_pred[0, t] = token_to_index[token]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_token = index_to_token[next_index+1] # dict starting at 1

            generated += [next_token]
            sentence = sentence[1:] + [next_token]

            sys.stdout.write(next_token)
            sys.stdout.flush()
        print()


def next_word_generative(model, sentence, index_to_token, max_len = 100):
    ## Generation of the next word
    last_char = ''
    word_generate = ''
    while last_char != ':':
        x_pred = np.zeros((1, max_len))
        for t, token in enumerate(sentence):
            if len(sentence) < max_len:
                x_pred[0, max_len - len(sentence) + t] = token_to_index[token]
            else:
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
            sentence = token_test if len(token_test) < step_t else token_test[-step_t:]
            word_i = next_word_generative(model, sentence, index_to_token)
            word_i_processed = word_i.replace("-", "")
            word_i_processed = word_i_processed.replace(":", "")
            ppl += np.log(perplexity_i(word_i_processed, words, corpus))
        else:
            words = corpus[start_index: start_index + step_t]
            token_test = get_processed_text(words, selectors)
            sentence = token_test if len(token_test) < step_t else token_test[-step_t:]
            word_i = next_word_generative(model, sentence, index_to_token)
            word_i_processed = word_i.replace("-", "")
            word_i_processed = word_i_processed.replace(":", "")
            ppl += np.log(perplexity_i(word_i_processed, words, corpus))
            start_index += 1

    return np.exp(-ppl/Ntest)


def perplexity_i(word_i_processed, words, corpus):
    indexes = km.kmpMatch(corpus, corpus)
    p_word = 0
    p_context = 0

    if len(indexes) == 0:
        return 0.01

    for i in indexes:
        p_context += 1
        if corpus[i:(i+len(words)+1)] == (words + word_i_processed):
            p_word += 1

    return p_word/p_context