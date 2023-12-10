import os
import pathlib
import tarfile
import urllib
import pandas as pd

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
