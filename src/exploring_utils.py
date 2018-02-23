from separadorSilabas import silabas


# Silabas

test = 0

if test == 1:
    word = 'atracción'
    s = 'a-trac-ción'
    answer = silabas(word)
    print(answer)
    if answer == s:
        print('OK!')

if test == 2:
    word = ' '
    answer = silabas(word)
    print(answer)

if test == 3:
    word = '-'
    answer = silabas(word)
    print(answer)

# add_space_file

from utils import add_space_file

path = '../data/test_exploring_utils.txt'
text = 'hola.'

with open(path, 'w') as f:
    f.write(text)

path_in = '../data/test_exploring_utils.txt'
path_out = '../data/test_exploring_utils_spaces.txt'

add_space_file(path_in = path_in,
               path_out = path_out)