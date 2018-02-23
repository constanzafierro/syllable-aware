# coding: utf-8
from separadorSilabas import silabas
from utils import preprocessing_file
from utils import *


test = 4


# Silabas

if test == 1:
    word = 'atracción'
    s = 'a-trac-ción'
    answer = silabas(word)
    print(answer)
    if answer == s:
        print('OK!')


if test == 2: #error
    word = ' '
    answer = silabas(word)
    print(answer)


if test == 3: #error
    word = '-'
    answer = silabas(word)
    print(answer)


# preprocesing_file

if test == 4:

    text = 'hola.' + '\ncómo estás?' + '\ntan helado que estai Juan!'
    s = 'hola' + ' . ' + '\ncómo estás' + ' ? ' + '\ntan helado que estai Juan' +' '*2 + '\n'
    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    path = '../test/data/test_exploring_utils.txt'
    with open(path, 'w') as f: f.write(text)

    path_in = '../test/data/test_exploring_utils.txt'
    path_out = '../test/data/test_exploring_utils_spaces.txt'

    preprocessing_file(path_in = path_in,
                       path_out = path_out,
                       to_ignore = to_ignore
                       )

    processed = open(path_out, 'r').read()

    if processed == s:
        print('OK!')


if test == 5: #error

    path_in = '../test/data/None.txt'
    path_out = '../test/data/test_exploring_utils_spaces.txt'

    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    preprocessing_file(path_in = path_in,
                       path_out = path_out,
                       to_ignore = to_ignore
                       )


if test == 6:

    word = 'palabra'
    s = ['pa-', 'la-', 'bra:']

    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':')
    print(answer)

    if answer == s:
        print('OK!')


if test == 7: #error

    word = 'atracción'
    s = ['a-', 'trac-', 'ción:']

    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':')
    if answer == s:
        print('OK!')


if test == 8: #error

    word = ' '
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':')