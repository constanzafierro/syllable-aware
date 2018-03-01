from keras.models import load_model
import numpy as np
import os
import json

path_to_file = "./log/evaluation_model.txt"

class evaluationModel():

    def __init__(self, model_file, tokenization_file, map_punctuation):

        if not os.path.exists(path= tokenization_file):
            raise FileNotFoundError("Path doesn't exists, '{}'".format(tokenization_file))

        with open(tokenization_file) as f:
            self.tokenization_params = json.load(f)

        if not os.path.exists(path= model_file):
            raise FileNotFoundError("Path doesn't exists, '{}'".format(model_file))

        self.model = load_model(model_file)

        self.map_punctuation = map_punctuation


    def predict_text(self,
                     seed = "declaro reanudado el período de sesiones",
                     nwords = 10,
                     temperature = 1.0):

        sentence = seed
        generated = sentence.copy()
        fprint("Generating with seed: {}".format(seed))
        words_count = 0
        while words_count < nwords:
            x_pred = np.zeros((1, self.tokenization_params["lprime"]))
            for t, token in enumerate(sentence):
                x_pred[0, t] = self.tokenization_params["token_to_index"][token]
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_token = self.tokenization_params["index_to_token"][next_index + 1]

            generated += [next_token]
            sentence = sentence[1:] + [next_token]

            if next_index in self.tokenization_params["index_ends"]:
                words_count += 1


        text_array = []
        for token in generated:
            if token in self.map_punctuation:
                text_array.append(" " + self.map_punctuation[token] + " ")
            else:
                text_array.append(token)


        text = "".join(text_array)
        text = text.replace(self.tokenization_params["final_char"], " ")
        text = text.replace(self.tokenization_params["inter_char"], "")

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