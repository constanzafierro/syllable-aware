# Data

## Syllable-aware tests

Spanish data for tests obtained from [HERE](https://github.com/yoonkim/lstm-char-cnn/blob/master/get_data.sh)


## Spanish Billion Words Corpus

Raw Data down obtained from [Spanish Billion Words Corpus](http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/clean_corpus.tar.bz2)

> Cristian Cardellino: Spanish Billion Words Corpus and Embeddings (March 2016), http://crscardellino.me/SBWCE/


## getData

```
wget http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/clean_corpus.tar.bz2

tar xf clean_corpus.tar.bz2

mv clean_corpus ./data/
```


# Conda Environment

## Create Environment (example name = venv1)

`conda create -n venv1 python=3.6 anaconda`

## Activate Environment
`source activate venv1`

## Install Tensorflow-GPU
`pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl`


