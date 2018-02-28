import numpy as np
import random

# TODO: We must set a seed !!


class GeneralGenerator():

    def __init__(self,
                 batch_size,
                 ind_tokens,
                 vocabulary,
                 max_len,
                 split_symbol_index,
                 count_to_split
                 ):

        '''Generates X,Y batches dynamically.
        Args:
            batch_size:
            ind_tokens:
            vocabulary:
            max_len:
            split_symbol_index: int with index for the symbol to split. If do not want a symbol to split, set to -1.
            count_to_split:
        '''

        #random.seed(779) # TODO: Seed

        self.ind_tokens = ind_tokens
        self.voc = vocabulary
        self.max_len = max_len
        self.batch_size = batch_size
        self.split_symbol_index = split_symbol_index
        self.count_to_split = count_to_split
        self.steps_per_epoch = int(len(ind_tokens) / batch_size) + 1


    def generator(self):

        n_features = len(self.voc)
        X_batch = np.zeros((self.batch_size, self.max_len), dtype = np.int32)
        Y_batch = np.zeros((self.batch_size, n_features), dtype = np.bool)

        current_batch_index = 0

        while True:
            left_limit = random.randint(0, len(self.ind_tokens) - self.max_len - 1)

            # find X data
            group_count = 0

            for i, e in enumerate(self.ind_tokens[left_limit:], left_limit):

                if i-left_limit+1 == self.max_len: # we achieved max len
                    break

                if e == self.split_symbol_index:

                    group_count += 1
                    if group_count == self.count_to_split: # we achieved max number of groups wanted
                        break

            right_limit = i

            # check minimum size
            if (right_limit - left_limit + 1) < int(self.max_len/2) + 2:
                continue

            # set X,Y data
            pad_length = (right_limit - left_limit + 1) - self.max_len
            for k, ind_token in enumerate([0]*pad_length + self.ind_tokens[left_limit:right_limit+1]):
                X_batch[current_batch_index, k] = ind_token
            Y_batch[current_batch_index, self.ind_tokens[right_limit+1]-1] = 1 # -1 because indexing starts in 1

            # check current_batch_index
            current_batch_index += 1

            if current_batch_index == self.batch_size:
                current_batch_index = 0

                yield X_batch, Y_batch

                X_batch = np.zeros((self.batch_size, self.max_len), dtype = np.int32)
                Y_batch = np.zeros((self.batch_size, n_features), dtype = np.bool)
