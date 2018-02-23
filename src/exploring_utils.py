# coding: utf-8
from separadorSilabas import silabas
from utils import *


test = 13


# Silabas

if test == 0: #error
    word = 'Ññ'
    answer = silabas(word)
    print(answer)


if test == 1:
    word = 'atracción'
    s = 'a-trac-ción'
    answer = silabas(word)
    assert answer == s



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
    assert processed == s


if test == 5: #error
    path_in = '../test/data/None.txt'
    path_out = '../test/data/test_exploring_utils_spaces.txt'

    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    preprocessing_file(path_in = path_in,
                       path_out = path_out,
                       to_ignore = to_ignore
                       )


if test == 6: #error
    word = 'Ññ'
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )
    print(answer)


if test == 7:
    word = 'palabra'
    s = ['pa-', 'la-', 'bra:']
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )
    assert answer == s


if test == 8: #error
    word = 'atracción'
    s = ['a-', 'trac-', 'ción:']
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )
    assert answer == s



if test == 9: #error
    word = ' '
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )

if test == 10:
    syllable = 'palabra'
    s = ['p-', 'a-', 'l-', 'a-', 'b-', 'r-', 'a-']
    answer = get_characters(syllable = syllable,
                            middle = '-',
                            end = ':'
                            )
    assert answer == s


if test == 11:
    syllable = 'pa-' + 'bra:'
    s = ['p-', 'a-', 'b-', 'r-', 'a:']
    answer = get_characters(syllable = syllable,
                            middle = '-',
                            end = ':'
                            )
    assert answer == s


if test == 12:
    syllable = ' ' + ' ' + ',.!' +'1'
    s = []
    answer = get_characters(syllable = syllable,
                            middle = '-',
                            end = ':'
                            )
    assert answer == s


if test == 13:
    freq_word = dict()
    word1 = 'hola'
    word2 = 'chabelalaila'
    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    get_freq_words(word = word1,
                   freq_word = freq_word,
                   to_ignore = to_ignore
                    )
    s1 = {'hola': 1}

    assert s1 == freq_word


    get_freq_words(word = word1,
                   freq_word = freq_word,
                   to_ignore = to_ignore
                   )
    s2 = {'hola': 2}
    assert s2 == freq_word

    get_freq_words(word = word2,
                   freq_word = freq_word,
                   to_ignore = to_ignore
                   )
    s3 = {'hola': 2, 'chabelalaila': 1}
    assert s3 == freq_word