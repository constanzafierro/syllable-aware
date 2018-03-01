import losswise
from losswise.libs import LosswiseKerasCallback

import keras
import json

# TODO : Inicializar correctamente para que el append sea safe
# TODO: Mover clase Callbacks al archivo RNN.py, renombrar archivo y actualizar los imports en otros archivos


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


    def losswise(self, keyfile, model_to_json, epochs, steps_per_epoch):

        keys = json.load(open(keyfile))

        api_key = keys["losswise_api_key"]
        tag = keys["losswise_tag"]

        losswise.set_api_key(api_key)

        params = json.loads(model_to_json)
        params['steps_per_epoch'] = steps_per_epoch
        params['epochs'] = epochs


        self.losswise_callback = LosswiseKerasCallback(tag=tag,
                                                       params=params
                                                       )


    def get_callbacks(self):

        self.callbacks.append(self.early_stopping)

        self.callbacks.append(self.checkpoint)

        self.callbacks.append(self.losswise_callback)

        return self.callbacks
