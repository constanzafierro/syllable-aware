import losswise
from src.callback_losswise import LosswiseKerasCallback

import keras
import json


class Callbacks:

    def __init__(self):

        self.callbacks = []


    def early_stopping(self, monitor, patience):

        self.early_stopping = keras.callbacks.EarlyStopping(monitor=monitor,
                                                            min_delta=0,
                                                            patience=patience,
                                                            verbose=0,
                                                            mode='auto'
                                                            )


    def checkpoint(self, filepath, monitor, save_best_only):

        self.checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                          monitor=monitor,
                                                          verbose=1,
                                                          save_best_only=save_best_only,
                                                          save_weights_only=False,
                                                          mode='auto',
                                                          period=1
                                                          )


    def losswise(self, keyfile, model_to_json, samples, steps, batch_size):

        keys = json.load(open(keyfile))

        api_key = keys["losswise_api_key"]
        tag = keys["losswise_tag"]

        losswise.set_api_key(api_key)

        params_data = json.loads(model_to_json)

        params_data['samples'] = samples
        params_data['steps'] = steps
        params_data['batch_size'] = batch_size

        params_model = {'batch_size': batch_size}

        self.losswise_callback = LosswiseKerasCallback(tag=tag,
                                                       params_data=params_data,
                                                       params_model=params_model
                                                       )

        self.losswise_callback.set_params(params=params_model)


    def get_callbacks(self):

        self.callbacks.append(self.early_stopping)

        self.callbacks.append(self.checkpoint)

        self.callbacks.append(self.losswise_callback)

        return self.callbacks
