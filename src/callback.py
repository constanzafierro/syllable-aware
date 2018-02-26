import keras
from losswise import Session
from keras import backend as K


class SilabasCallback(keras.callbacks.Callback):

    def __init__(self, tag=None, params={}):
        # model hyper parameters, json serializable Python object
        self.tag = tag
        self.params = params

        self.graph_map = {}
        self.metrics = []
        self.x = 0

    def on_train_begin(self, logs={}):

        # necessary attributes to losswise
        ## Con esto Losswise estima cuanto tiempo se demorará
        self.max_iter = int(self.params['epochs'] * self.params['steps_per_epoch'] / self.params['batch_size'] + 1)
        self.session = Session(tag=self.tag, max_iter=self.max_iter, params=self.params)

        for metric in self.params['metrics']:
            if not metric.startswith('val_'):
                if metric not in self.metrics:
                    self.metrics.append(metric)

        for metric in self.metrics:
            if 'acc' in metric:
                kind = 'max'
            else:
                kind = 'min'
            self.graph_map[metric] = self.session.graph(metric, kind=kind)

        self.x = 0

        return

    def on_train_end(self, logs={}):
        # required by losswise
        ## Con esto losswise sabe que el programa terminó, si no lo recibe piensa que falló y cambia
        ## la etiqueta a Cancelled !!
        self.session.done()
        return

    def on_epoch_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        ## Validation !!
        for metric in self.metric_val:
            if metric in logs:
                data = {metric: logs[metric]}
                self.graph_map[metric].append(self.x, data)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        for metric in self.metric_list:
            data = {metric: logs.get(metric)}
            self.graph_map[metric].append(self.x, data)
        self.x += 1
        return


def metric_pp(average_TPW, cross_entropy):

    def perplexity(y_true, y_pred):

        #cross_entropy = K.categorical_crossentropy(y_true, y_pred)
        bpt = average_TPW * cross_entropy

        pp = K.pow(2.0, bpt)

        return pp

    return perplexity