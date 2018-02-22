

def main():
    flag = False

    path_in = './data/icm14_es/train.txt'
    path_out = './data/icm14_es/train_add_space.txt'

    to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&'''

    if flag:
        preprocessing_file(path_in, path_out, to_ignore)

    sign_to_ignore = [i for i in to_ignore]
    tokenSelector = TokenSelector(sign_to_ignore = sign_to_ignore)
    tokenSelector.get_dictionary(path_out)

    tw = [30, 60, 100, 300, 600, 1000, 3000, 6000]
    totalt = 10000
    sequence_length = [100, 500, 1000, 2500, 5000]

    print('='*50)
    print('corpus to proccess : {}'.format(path_in))
    print('vocabulary  word size = {} \t vocabulary syllables size = {}'.format(len(tokenSelector.dict_word), len(tokenSelector.dict_syll)))

    for sl in sequence_length:

        print('='*50)
        print('sequence length = {}'.format(sl))

        for t in tw:
            q_word = t
            q_syll = totalt - t
            tokenSelector.get_frequent(quantity_word = q_word, quantity_syll = q_syll)
            #print(tokenSelector.syllables)

            token_selected = []
            with open(path_out) as f1:
                for line in f1:
                    words = line.lower().split()
                    for token in words:
                        token = token.strip()
                        tokenSelector.select(token, token_selected)

            print('number of words = {} \t number of syllables = {} \t Lprime = {}'.format(q_word, q_syll, Lprime(token_selected,sl) ))


if __name__ == '__main__':
    main()
