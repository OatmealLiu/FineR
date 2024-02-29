import os
import json
import pandas as pd
import errno
import csv
import numpy as np
import shutil
from copy import deepcopy


def mkdir_if_missing(directory: str):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def dump_json(filename: str, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")


def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def dump_txt(filename: str, in_data: str):
    if not filename.endswith('.txt'):
        filename += '.txt'

    with open(filename, 'w') as fbj:
        fbj.write(in_data)


def load_txt(filename: str):
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as fbj:
        return fbj.read()


def write_csv_col(filename: str, in_data):
    df = pd.DataFrame(in_data)
    df.to_csv(filename, index=False)


