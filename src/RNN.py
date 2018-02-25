import keras


class RecurrentLSTM:

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 input_length,
                 recurrent_dropout,
                 dropout,
                 seed
                 ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_length = input_length
        self.recurrent_dropout = recurrent_dropout
        self.dropout = dropout
        self.seed = seed


        self.word_embeddings = keras.layers.Embedding(input_dim=self.vocab_size + 1,
                                                      output_dim=self.embedding_dim,
                                                      input_length=self.input_length,
                                                      mask_zero=True
                                                      )

        self.lstm_1 = keras.layers.LSTM(units=self.hidden_dim,
                                        recurrent_dropout=self.recurrent_dropout,
                                        return_sequences=True,
                                        unroll=False,
                                        implementation=2
                                        )

        self.dropout_1 = keras.layers.Dropout(rate=self.dropout,
                                              seed=self.seed
                                              )

        self.lstm_2 = keras.layers.LSTM(units=self.hidden_dim,
                                        recurrent_dropout=self.recurrent_dropout,
                                        return_sequences=False,
                                        unroll=False,
                                        implementation=2
                                        )

        self.dense = keras.layers.Dense(units=self.vocab_size,
                                        activation='softmax'
                                        )


    def build(self, optimizer, metrics):

        self.optimizer = optimizer
        self.metrics = metrics

        # self.learning_rate = learning_rate # (add to forward)
        # self.optimizer = keras.optimizers.RMSprop(lr = self.learning_rate)


        # Build

        self.model = keras.models.Sequential([self.word_embeddings, self.lstm_1, self.dropout_1, self.lstm_2, self.dense])

        self.summary = self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=self.metrics
                           )


    def fit(self, train_generator, val_generator, epochs, callbacks, workers):

        self.train_generator = train_generator # Object/Instance Generator, containing .generator() and .steps_per_epoch
        self.val_generator = val_generator # Object/Instance Generator, containing .generator() and .steps_per_epoch

        self.epochs = epochs
        self.workers = workers
        self.callbacks = callbacks

        # TODO: Agregar val_generator al fit_generator y al input del fit()
        # https://keras.io/models/sequential/#fit_generator
        self.model.fit_generator(generator=self.train_generator.generator(),
                                 steps_per_epoch=self.train_generator.steps_per_epoch,
                                 epochs=self.epochs,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 validation_data= None# self.val_generator.generator(), # TODO: (1)
                                 validation_steps=None#1 # TODO: (2)
                                 class_weight=None,
                                 max_queue_size=10,
                                 workers=self.workers,
                                 use_multiprocessing=False, # Must be False, unless there is a "thread safe generator"
                                 shuffle=False, # Must be False
                                 initial_epoch=0 # TODO: Ver si es útil añadirlo como input del fit()
                                 )

    def summary(self):

        return self.summary
