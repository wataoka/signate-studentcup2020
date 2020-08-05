import os
import sys

import numpy as np
import pandas as pd

sys.path.append('..')
from constants import BASE_DIR, DATASET_DIR

def load_dataset():
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    submit_sample = pd.read_csv(os.path.join(DATASET_DIR, 'submit_sample.csv'))
    return train, test, submit_sample