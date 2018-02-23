# coding: utf-8
from separadorSilabas import silabas
from utils import preprocessing_file, get_syllables, get_characters, get_freq_words, word_to_syll, syll_to_charac

test = 18


# silabas

if test == 0: #error
    word = 'Ññ'
    answer = silabas(word, sep='-')
    print(answer)


if test == 1:
    word = 'palabra'
    s = 'pa-la-bra'
    answer = silabas(word, sep='-')
    assert answer == s


if test == 2:
    word = 'atracción'
    s = 'a-trac-ción'
    answer = silabas(word, sep='-')
    assert answer == s



if test == 3: #error
    word = ' '
    answer = silabas(word, sep='-')
    print(answer)


if test == 4: #error
    word = '-'
    answer = silabas(word, sep='-')
    print(answer)


# preprocesing_file

if test == 5:
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


if test == 6: #error
    path_in = '../test/data/None.txt'
    path_out = '../test/data/test_exploring_utils_spaces.txt'

    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    preprocessing_file(path_in = path_in,
                       path_out = path_out,
                       to_ignore = to_ignore
                       )


# get_syllables

if test == 7: #error
    word = 'Ññ'
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )
    print(answer)


if test == 8:
    word = 'palabra'
    s = ['pa-', 'la-', 'bra:']
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )
    assert answer == s


if test == 9: #error
    word = 'atracción'
    s = ['a-', 'trac-', 'ción:']
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )
    assert answer == s



if test == 10: #error
    word = ' '
    answer = get_syllables(word = word,
                           middle = '-',
                           end = ':'
                           )


# get_characters

if test == 11:
    syllable = 'palabra'
    s = ['p-', 'a-', 'l-', 'a-', 'b-', 'r-', 'a-']
    answer = get_characters(syllable = syllable,
                            middle = '-',
                            end = ':'
                            )
    assert answer == s


if test == 12:
    syllable = 'pa-' + 'bra:'
    s = ['p-', 'a-', 'b-', 'r-', 'a:']
    answer = get_characters(syllable = syllable,
                            middle = '-',
                            end = ':'
                            )
    assert answer == s


if test == 13:
    syllable = ' ' + ' ' + ',.!' +'1'
    s = []
    answer = get_characters(syllable = syllable,
                            middle = '-',
                            end = ':'
                            )
    assert answer == s


# get_freq_words

if test == 14:
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


# word_to_syll

if test == 15:
    dict_word = dict()

    word1 = 'hola'
    word2 = 'chabelalaila'
    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    word_to_syll(word = word1,
                 dict_word = dict_word,
                 to_ignore = to_ignore,
                 middle = '-',
                 end = ':',
                 sign_not_syllable = '<sns>',
                 verbose = False
                 )
    s1 = {'hola': ['ho-', 'la:']}
    assert s1 == dict_word

    word_to_syll(word = word2,
                 dict_word = dict_word,
                 to_ignore = to_ignore,
                 middle = '-',
                 end = ':',
                 sign_not_syllable = '<sns>',
                 verbose = False
                 )
    s2 = {'hola': ['ho-', 'la:'], 'chabelalaila': ['cha-', 'be-', 'la-', 'lai-', 'la:']}
    assert s2 == dict_word


if test == 16:
    word= 'palabra'
    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    dict_word = dict()
    word_to_syll(word = word,
                 dict_word = dict_word,
                 to_ignore = to_ignore,
                 middle = '-',
                 end = ':',
                 sign_not_syllable = '<sns>',
                 verbose = False
                 )
    s = {'palabra': ['pa-', 'la-', 'bra:']}
    assert s == dict_word


if test == 17:
    word= 'atracción'
    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    dict_word = dict()
    word_to_syll(word = word,
                 dict_word = dict_word,
                 to_ignore = to_ignore,
                 middle = '-',
                 end = ':',
                 sign_not_syllable = '<sns>',
                 verbose = False
                 )
    s = {'atracción': ['a-', 'trac-', 'ción:']}
    assert s == dict_word


# syll_to_charac

if test == 18:

    word1 = 'hola'
    word2 = 'chabelalaila'

    dict_syll = dict()
    dict_word = {'hola':['ho-', 'la-'], 'chabelalaila':['cha-','be-','la-','lai-','la:']}


    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    dict_syll = syll_to_charac(word = word1,
                   dict_syll = dict_syll,
                   dict_word = dict_word,
                   to_ignore = to_ignore,
                   middle = '-',
                   end = ':',
                   sign_not_syllable = '<sns>'
                   )

    dict_syll = syll_to_charac(word = word2,
                   dict_syll = dict_syll,
                   dict_word = dict_word,
                   to_ignore = to_ignore,
                   middle = '-',
                   end = ':',
                   sign_not_syllable = '<sns>'
                   )

    print(dict_word)
    print(dict_syll)