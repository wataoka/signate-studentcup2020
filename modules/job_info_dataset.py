import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

sys.path.append('..')
from constants import BASE_DIR, DATASET_DIR


class JobInfoDataset(Dataset):
    
    def __init__(self, X, y=None, jobflag=None):
        self.X = X
        self.y = y
        self.jobflag = jobflag

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]
