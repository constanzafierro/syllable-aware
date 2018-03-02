from keras import backend as K

# https://datascience.stackexchange.com/a/19411

def metric_pp(average_TPW):

    def perplexity(y_true, y_pred):
        cross_entropy = K.categorical_crossentropy(y_true, y_pred)

        bpt = average_TPW * cross_entropy

        pp = K.pow(2.0, bpt)

        return pp

    return perplexity
