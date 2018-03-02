from src.utils import preprocessing_file
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile', help='path to infile')
parser.add_argument('-o','--outfile', help='path to outfile')

args = parser.parse_args()

if not args.ifile:
    raise ValueError("missing argument --ifile")

if not args.ofile:
    raise ValueError("missing argument --ofile")

path_in = args.ifile
path_out = args.ofile
to_ignore = '''¡!()[]{}\"\'0123456789…-=@+*\t%&//­\xc2'''

punctuation =  {'¿': '<ai>',
                '?': '<ci>',
                '.': '<pt>',
                '\n': '<nl>',
                ',': '<cm>',
                '<unk>': '<unk>',
                ':': '<dc>',
                ';': '<sc>'
                }

token_unknow = '<unk>'

preprocessing_file(path_in, path_out, to_ignore, punctuation, token_unknow)