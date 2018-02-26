from losswise import Session
from keras.callbacks import Callback


class LosswiseKerasCallback(Callback):

    def __init__(self, tag=None, params={}):
    #def __init__(self, tag=None, params_data={}, params_model={}):

        # model hyper parameters, json serializable Python object
        self.tag = tag
        self.params_data = params
        #self.params_data = params_data
        #self.params_model = params_model
        self.graph_map = {}
        #self.params = {}

        super(LosswiseKerasCallback, self).__init__()

    # set_params se hereda de keras.kallbacks.Callback
    #def set_params(self, params):
    #    self.params = params

    def on_train_begin(self, logs={}):
        self.max_iter = int(self.params['epochs'] * self.params['steps'] / self.params['batch_size'] + 1)
        #self.max_iter = int(self.params['epochs'] * self.params['steps'] / self.params_model['batch_size'] + 1)
        self.session = Session(tag=self.tag, max_iter=self.max_iter, params=self.params_data)
        self.metric_list = []
        for metric in self.params['metrics']:
            if not metric.startswith('val_'):
                if metric not in self.metric_list:
                    self.metric_list.append(metric)
        for metric in self.metric_list:
            if 'acc' in metric:
                kind = 'max'
            else:
                kind = 'min'
            self.graph_map[metric] = self.session.graph(metric, kind=kind)
        self.x = 0
    def on_epoch_end(self, epoch, logs={}):
        for metric in self.metric_list:
            metric_val = "val_" + metric
            if metric_val in logs:
                data = {metric_val: logs[metric_val]}
                self.graph_map[metric].append(self.x, data)
    def on_batch_end(self, batch, logs={}):
        for metric in self.metric_list:
            data = {metric: logs.get(metric)}
            self.graph_map[metric].append(self.x, data)
        self.x += 1
    def on_train_end(self, logs={}):
        self.session.done()