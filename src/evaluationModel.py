from keras.models import load_model
from src.TokenSelector import TokenSelector

import numpy as np
import os
import json

path_to_file = "./log/evaluation_model.txt"

class evaluationModel():

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


    def predict_text(self,
                     seed = "declaro reanudado el período de sesiones",
                     nwords = 10,
                     temperature = 1.0):

        seed = seed.lower().split()

        sentence = []

        for word in seed:
            sentence = self.tokenSelector.select(word, sentence)

        generated = (sentence + ["."])[:-1]

        fprint("Generating with seed: {}".format(seed))
        words_count = 0
        while words_count < nwords:
            x_pred = np.zeros((1, self.tokenization_params["lprime"]))
            for t, token in enumerate(sentence):
                x_pred[0, t] = self.tokenization_params["token_to_index"][token]
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_token = self.tokenization_params["index_to_token"][str(next_index + 1)]

            generated += [next_token]
            sentence = sentence[1:] + [next_token]

            if next_index in self.tokenization_params["index_ends"]:
                words_count += 1


        text_array = []
        for token in generated:
            if token in self.map_punctuation_inv:
                text_array.append(" " + self.map_punctuation_inv[token] + " ")
            else:
                text_array.append(token)

        text = "".join(text_array)
        text = text.replace(self.tokenSelector_params["final_char"], " ")
        text = text.replace(self.tokenSelector_params["inter_char"], "")

        fprint(text)








def fprint(text):
    print(text)
    with open(path_to_file, "a") as out:
        out.write(text)


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