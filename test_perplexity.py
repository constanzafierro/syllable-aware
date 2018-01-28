import unittest
from perplexity import *

class TestPerplexityMethods(unittest.TestCase):

    def test_conditional_prob_wordi(self):
        corpus = [i for i in 'ABCACACABACBABCACBABC']
        
        self.assertEqual(conditional_prob_wordi('C', ['A','B'], corpus),0.75)
        self.assertEqual(conditional_prob_wordi('B', ['A'], corpus),0.5)
        self.assertEqual(conditional_prob_wordi('D', ['A','B'], corpus),0)

        corpus = [i for i in 'hola como estas hola bien y tu'.split()]
        self.assertEqual(conditional_prob_wordi('como', ['hola'], corpus),0.5)
        self.assertEqual(conditional_prob_wordi('bien', ['hola'], corpus),0.5)
        self.assertEqual(conditional_prob_wordi('estas', ['como'], corpus),1)
        self.assertEqual(conditional_prob_wordi('estas', ['hola'], corpus),0)


if __name__ == '__main__':
    unittest.main()
