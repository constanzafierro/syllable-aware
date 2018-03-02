from keras.models import load_model
from src.TokenSelector import TokenSelector
from src.Tokenization import Tokenization

import numpy as np
import os
import json


class CheckModel():

    def __init__(self, model_path, tokenization_path, tokenSelector_path):

        raisePathNotExists(model_path)
        raisePathNotExists(tokenization_path)
        raisePathNotExists(tokenSelector_path)

        with open(tokenization_path) as f:
            self.tokenization_params = json.load(f)

        self.model = load_model(model_path)

        with open(tokenSelector_path) as f:
            self.tokenSelector_params = json.load(f)

        self.tokenSelector = TokenSelector(final_char= self.tokenSelector_params["final_char"],
                                           inter_char= self.tokenSelector_params["inter_char"],
                                           signs_to_ignore= self.tokenSelector_params["signs_to_ignore"],
                                           words_to_ignore= self.tokenSelector_params["words_to_ignore"],
                                           map_punctuation= self.tokenSelector_params["map_punctuation"],
                                           letters= "".join(self.tokenSelector_params["letters"]),
                                           sign_not_syllable= self.tokenSelector_params["sign_not_syllable"]
                                           )

        self.tokenSelector.set_params(self.tokenSelector_params)

        self.map_punctuation_inv = dict()
        for key, value in self.tokenSelector_params["map_punctuation"]:
            self.map_punctuation_inv[value] = key

    def predict_word(self, sentence, temperature):

        next_token = ""
        generated = []
        while next_token not in self.tokenSelector_params["index_ends"]:
            x_pred = np.zeros((1, self.tokenization_params["lprime"]))
            for t, token in enumerate(sentence):
                x_pred[0, t] = self.tokenization_params["token_to_index"][token]
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_token = self.tokenization_params["index_to_token"][str(next_index + 1)]
            generated += [next_token]
            sentence = sentence[1:] + [next_token]

        word = ""
        for token in generated:
            if token in self.map_punctuation_inv:
                word += self.map_punctuation_inv[token]+ " "
            else:
                token = token.replace(self.tokenSelector_params["final_char"], " ")
                token = token.replace(self.tokenSelector_params["inter_char"], "")
                word += token

        return word, generated


    def predict_text(self,
                     seed = "declaro",
                     nwords = 10,
                     temperature = 1.0):

        seed_array = seed.lower().split()

        sentence = []
        text = ""

        for word in seed_array:
            sentence = self.tokenSelector.select(word, sentence)

        for i in range(nwords):
            word, generated = self.predict_word(sentence, temperature)
            text += word
            sentence += generated

        return text



    def get_probability_token(self, token, sentence):
        x_pred = np.zeros((1, self.tokenization_params["lprime"]))
        for t, tk in enumerate(sentence):
            x_pred[0, t] = self.tokenization_params["token_to_index"][tk]
        preds = self.model.predict(x_pred, verbose=0)[0]

        return preds[token]

    def perplexity(self, path_to_test):

        tokenization = Tokenization(path_to_file=None,
                                    final_char=self.tokenSelector_params["final_char"],
                                    final_punc=self.tokenSelector_params["final_punc"],
                                    inter_char=self.tokenSelector_params["inter_char"],
                                    signs_to_ignore=self.tokenSelector_params["signs_to_ignore"],
                                    words_to_ignore=self.tokenSelector_params["words_to_ignore"],
                                    map_punctuation=self.tokenSelector_params["map_punctuation"],
                                    letters=self.tokenSelector_params["letters"],
                                    sign_not_syllable=self.tokenSelector_params["sign_not_syllable"],
                                    )
        tokenization.set_params_experiment(self.tokenization_params)
        max_len = self.tokenization_params["lprime"]
        test_selected = tokenization.select_tokens(path_to_test)

        sentence = []
        token_word = []
        ppl = 0
        qw = 0

        for i,token in enumerate(test_selected):
            token_word += token
            if token[-1] in [self.tokenSelector_params["final_char"], self.tokenSelector_params["final_punc"]]:
                if len(sentence) < 1:
                    sentence += token_word
                    token_word = []
                    continue

                seed = sentence
                probs_word = 1
                ## Calculo la probabilidad de la palabra
                for tok in token_word:
                    probs_word *= self.get_probability_token(tok, seed)
                    seed += [tok]

                ## sumo la probabilidad a la del resto
                ppl += np.log2(probs_word)

                sentence += token_word
                qw += 1
                token_word = []
                if len(sentence) > max_len:
                    sentence = sentence[-max_len:]

        return - ppl / qw




def sample(pred, temperature=1.0):
    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    prob = np.random.multinomial(1, pred, 1)

    return np.argmax(prob)


def raisePathNotExists(path_to_file):
    if not os.path.exists(path= path_to_file):
        raise FileNotFoundError("Path doesn't exists, '{}'".format(path_to_file))