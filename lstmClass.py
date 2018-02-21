import keras

class Model:
  
  def __init__(self,
               vocab_size,
               embedding_dim,
               hidden_dim,
               input_length,
               recurrent_dropout,
               dropout,
               seed):

    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.input_length = input_length
    self.recurrent_dropout = recurrent_dropout
    self.dropout = dropout
    self.seed = seed


    self.word_embeddings = keras.layers.Embedding(input_dim = self.vocab_size+1,
                                                  output_dim = self.embedding_dim,
                                                  input_length = self.input_length,
                                                  mask_zero = True)

    self.lstm_1 = keras.layers.LSTM(units = self.hidden_dim,
                                    recurrent_dropout = self.recurrent_dropout,
                                    return_sequences = True,
                                    unroll = False,
                                    implementation = 2)

    self.dropout_1 = keras.layers.Dropout(rate = self.dropout,
                                          seed = self.seed)

    self.lstm_2 = keras.layers.LSTM(units = self.hidden_dim,
                                    recurrent_dropout = self.recurrent_dropout,
                                    return_sequences = False,
                                    unroll = False,
                                    implementation = 2)

    self.dense = keras.layers.Dense(units = self.vocab_size,
                                    activation = 'softmax')
    
    
    
  def build(self, optimizer, metrics):   
    
    self.optimizer = optimizer    
    self.metrics = metrics
    
    # self.learning_rate = learning_rate # (add to forward)
    # self.optimizer = keras.optimizers.RMSprop(lr = self.learning_rate)
    
    
    # Build
    
    self.model = keras.models.Sequential([self.word_embeddings, self.lstm_1, self.dropout_1, self.lstm_2, self.dense])
    
    self.summary = self.model.summary()
    
    self.model.compile(loss = 'categorical_crossentropy',
                       optimizer = self.optimizer,
                       metrics = self.metrics)
    
    #return self.model
  
  
  def fit(self, generator, epochs, workers, callbacks):
    
    self.g = generator # Object/Instance Generator, containing .generator() and .steps_per_epoch
    
    self.epochs = epochs
    self.workers = workers  
    self.callbacks = callbacks

    self.model.fit_generator(generator = self.g.generator(),
                             steps_per_epoch = self.g.steps_per_epoch,
                             epochs= self.epochs,
                             workers = self.workers,
                             callbacks = self.callbacks,
                             shuffle = False)
    
    
  def summary(self):    
    return self.summary
