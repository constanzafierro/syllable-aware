import unittest
import sys

class TestUtils(unittest.TestCase):

    def test_metricpp(self):

        avg_tpw = 5.0

        y_pred = K.constant([0.8, 0.2])
        y_true = K.constant([0.7, 0.3])

        perplexity_predict = metric_pp(avg_tpw)
        pp_predict = perplexity_predict(y_true, y_pred)

        cross_entropy_true = K.categorical_crossentropy(y_true, y_pred)
        bpt_true = avg_tpw * cross_entropy_true
        pp_true = K.pow(2.0, bpt_true)

        if K.backend() == 'tensorflow':
            tf_session = K.get_session()
            pp_true_eval = pp_true.eval(session=tf_session)
            pp_predict_eval = pp_predict.eval(session=tf_session)
        else:
            pp_true_eval = pp_true.eval()
            pp_predict_eval = pp_predict.eval()

        self.assertEqual(pp_predict_eval, pp_true_eval)


if __name__ == '__main__':

    sys.path.append("..")

    from keras import backend as K
    from src.perplexity import metric_pp

    unittest.main()