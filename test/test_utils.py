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

        word = ' '
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')
        self.assertTrue('No se reconoce el carácter' in str(context.exception))

        word = '-'
        with self.assertRaises(TypeError) as context:
            silabas(word, sep='-')
        self.assertTrue('No se reconoce el carácter' in str(context.exception))

    def test_silabas(self):
        word = 'palabra'
        syll_true = 'pa-la-bra'
        answer = silabas(word, sep='-')
        self.assertEqual(syll_true, answer)

        word = 'paralelepipedo'
        syll_true = 'pa-ra-le-le-pi-pe-do'
        answer = silabas(word, sep='-')
        self.assertEqual(syll_true, answer)

        word = 'atención'
        syll_true = 'a-ten-ción'
        answer = silabas(word, sep='-')
        self.assertEqual(syll_true, answer)

        word = 'atracción'
        syll_true = 'a-trac-ción'
        answer = silabas(word, sep='-')
        self.assertEqual(syll_true, answer)

    def test_preprocessing(self):
        text = 'hola.' + '\ncómo estás?' + '\ntan helado que estai Juan!'
        s = 'hola' + ' . ' + '\ncómo estás' + ' ? ' + '\ntan helado que estai Juan' + ' ' * 2 + '\n'
        to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

        path = './data/test_exploring_utils.txt'
        with open(path, 'w') as f: f.write(text)

        path_in = './data/test_exploring_utils.txt'
        path_out = './data/test_exploring_utils_spaces.txt'

        preprocessing_file(path_in=path_in,
                           path_out=path_out,
                           to_ignore=to_ignore
                           )

        processed = open(path_out, 'r').read()
        self.assertEqual(s, processed)

    def test_preprocessing_file_error(self):
        path_in = './data/None.txt'
        path_out = './data/test_exploring_utils_spaces.txt'

        to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''


        with self.assertRaises(TypeError) as context:
            preprocessing_file(path_in=path_in,
                               path_out=path_out,
                               to_ignore=to_ignore
                               )


        self.assertTrue('File not exists' in str(context.exception))


if __name__ == '__main__':
    sys.path.append("..")
    from src.separadorSilabas import silabas
    from src.utils import preprocessing_file
    unittest.main()