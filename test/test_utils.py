import unittest
import sys


class TestUtils(unittest.TestCase):
    
    def test_silabas_raise(self):
        word = 'Ññ'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')
        self.assertTrue('Estructura de sílaba incorrecta en la palabra' in str(context.exception))

        word = 'specific'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')
        self.assertTrue('Estructura de sílaba incorrecta en la palabra' in str(context.exception))

        word = 'status'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')
        self.assertTrue('Estructura de sílaba incorrecta en la palabra' in str(context.exception))

#    def test_silabas_


if __name__ == '__main__':
    sys.path.append("..")
    from src.separadorSilabas import silabas
    unittest.main()