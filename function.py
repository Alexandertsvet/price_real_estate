import os
import pathlib
import tarfile
import urllib
import pandas as pd
import numpy as np

np.random.seed(42)

DOWNLOAD_FILE = 'https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz'

BASE_DIR = pathlib.Path.cwd()
FILE_NAME = 'housing.tgz'
FILE_NAME_CSV = 'housing.csv'

FILE_DIR = BASE_DIR/'housing.tgz'


def load_dataset(url=DOWNLOAD_FILE, dir=FILE_DIR):
    urllib.request.urlretrieve(url, FILE_NAME)
    file = tarfile.open(dir)
    file.extractall(path=BASE_DIR)
    file.close()


def load_csv(dir=BASE_DIR/FILE_NAME_CSV):
    return pd.read_csv(dir)


def split_train_test(data, test_size):
    shuffle_index = np.random.permutation(len(data))
    border = int(len(data)*test_size)
    train_index = shuffle_index[border:]
    test_index = shuffle_index[:border]
    return data.iloc[train_index],data.iloc[test_index]
