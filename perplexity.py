import kmp as km
import numpy as np
from process_text import *
from string import punctuation


def strip_punctuation(sting):
    return ''.join(c for c in string if c not in punctuation)

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


def next_word_generate(model, sentence, index_to_token, token_to_index, max_len = 100):
    '''Generate of the next word like a string
    Args:
        model: is a pre-trained model
        sentence: array of string, is the input of the model to predict the next word
        index_to_token: dictionary of index (int) to token
        token_to_index: dictionary of token to index (int)
        max_len: dimension of max input to model, by default max_len=100
    Returns:
        Array with string tokens
    '''
    ## Generate of the next word
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

    #print('palabra generada--- {}'.format(" ".join(word_generate)))
    return word_generate.replace(':','').replace('.','')


def get_array_words(corpus, selectors):
    tokens = get_processed_text(corpus, selectors)
    words = []
    for token in tokens:
        words.append(token.replace(':','').replace('-',''))
    return words


def test_eval(model, corpus, selectors, step_t = 3):
    '''Evaluation of model with perplexity like metrics
    Args:
        model: is a pre-trained model
        corpus: array of string to eval de model
        selectors: array of tokens to tokenize corpus
        step_t: size of windows to eval perplexity under markov assumption
    Returns:
        Array with string tokens
    '''

    corpus = strip_punctuation(corpus)
    only_word = get_selectors(corpus, quantity_word = 1.0, quantity_syllable = 0.0)
    Ntest = len(get_processed_text(corpus, only_word))

    token_test = get_processed_text(corpus, selectors)

    words_array = get_array_words(corpus, only_word)

    # crear diccionario tokens-int
    string_voc = set(token_test)
    token_to_index = dict((t, i) for i, t in enumerate(string_voc, 1))
    index_to_token = dict((token_to_index[t], t) for t in string_voc)

    start_index = 0
    ppl = 0
    for i in range(1, len(words_array) - step_t - 1):
        #start_index
        if i < step_t:
            words = words_array[start_index: start_index + i + 1]
            token_test = get_processed_text(' '.join(words), selectors)
            sentence = token_test if len(token_test) < step_t else token_test[-step_t:]
            word_i = next_word_generate(model, sentence, index_to_token, token_to_index)
            ppl += np.log(conditional_prob_wordi(word_i, words, corpus))
        else:
            words = words_array[start_index: start_index + step_t]
            token_test = get_processed_text(' '.join(words), selectors)
            sentence = token_test if len(token_test) < step_t else token_test[-step_t:]
            word_i = next_word_generate(model, sentence, index_to_token, token_to_index)
            ppl += np.log(conditional_prob_wordi(word_i, words, corpus))
            start_index += 1

    return -ppl/Ntest


def conditional_prob_wordi(word_i_processed, words, corpus):
    '''estimate the conditional probability of a word in a given context a corpus
       P(word_i_processed | words)
    Args:
        word_i_processed: string to be evaluate 
        words: array of words before to word_i_processed
        corpus: array of words
    Returns:
         float, P(word_i_processed | words) = count(word_i_processed-words)/counts(words) in corpus
    '''
    indexes = km.kmpMatch(words, corpus)
    p_word = 0
    p_context = 0

    if len(indexes) == 0:
        return 0.001

    for i in indexes:
        p_context += 1
        if corpus[i:(i+len(words)+1)] == (words[0:] + [word_i_processed]):
            p_word += 1

    return p_word/p_context