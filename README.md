# TODO list

### Clase TokenSelector

*[DONE]* Crear Clase TokenSelector (Extraer class TokenSelector() desde process_corpus.py)


### Clase utils

*[DONE]* Crear Clase utils.py con las funciones de preprocess_corpus.py.


### Modificar "imports" en Encabezados

Modificar encabezados de los archivos para incluir clases utils.py y TokenSelector.py en los imports


### Crear main

Crear main() (Extraer class main() desde process_corpus.py )


### perplexity.py

Modificar perplexity.py para incorporar Clases TokenSelector.py, lstmClass.py

Editar perplexity.py y eliminar métodos no utilizados ( o guardarlos momentaneamente en un archivo txt )

Mover a clase utils.py los métodos no relacionados con el cálculo de la perplexity







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

mv spanish_billion_words ./data/
```


# Conda Environment


## Create Environment (example name = venv1)

```
conda create -n venv1 python=3.6 theano=1.0 anaconda
```

## Activate Environment

```
source activate venv1
```

## Theano Install

```
pip install theano --upgrade
```

## Keras Install

```
pip install keras --upgrade
```

## Switch KERAS Backend to Theano

Keras Documentation: [Switching from one backend to another](https://keras.io/backend/#switching-from-one-backend-to-another)


```
cd ~/.keras/
nano keras.json

```

Edit keras.json, and change the field "backend" to "theano"

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

---


## Install Tensorflow-GPU 1.4.0 (because there is CUDA Toolkit 8.0)

```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
```

---

## Extras

```
conda install -n venv1 mkl
conda install -n venv1 mkl-service
conda install -n venv1 -c conda-forge blas
conda install -n venv1 -c anaconda pygpu
```


