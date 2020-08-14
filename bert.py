import os
import sys
import time
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import torch
from simpletransformers.classification import ClassificationModel

# functions
def metric_f1(labels, preds):
    return f1_score(labels, preds, average='macro')

def seed_everything(seed):
    """for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # config
    SEED = 2020
    BASE_DIR = 'drive/My Drive/compe/signate-studentcup2020/'
    DATASET_DIR = BASE_DIR+'dataset/'
    SUBMITS_DIR = BASE_DIR+'submits/'
    TEXT_COL = "description"
    TARGET = "jobflag"
    NUM_CLASS = 4
    N_FOLDS = 4

    num_for_labels = [624, 348, 1376, 583]
    total = sum(num_for_labels)
    weight = []
    for num in num_for_labels:
        weight.append((total-num)/num)

    # fix seed
    seed_everything(SEED)

    # preprocessing
    train = pd.read_csv(DATASET_DIR+"train.csv").drop(['id'], axis=1)
    train = train.rename(columns={TEXT_COL:'text', TARGET:'label'})
    train['label'] -= 1

    test = pd.read_csv(DATASET_DIR+"test.csv")
    test = test.rename(columns={TEXT_COL:'text'}).drop(['id'], axis=1)

    # training
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train['fold_id'] = -1
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train.index, train['label'])):
        train.loc[train.iloc[valid_idx].index, 'fold_id'] = fold

    X_train = train.loc[train['fold_id']!=0]
    X_valid = train.loc[train['fold_id']==0]

    params = {
        "overwrite_output_dir": "outputs/",
        "max_seq_length": 128,
        "train_batch_size": 64,
        "eval_batch_size": 32,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "manual_seed":SEED,
    }
    model = ClassificationModel('bert',
                                'bert-base-cased',
                                num_labels=4,
                                weight=weight,
                                args=params,
                                use_cuda=True)
    
    model.train_model(X_train)

    result, model_outputs, wrong_predictions = model.eval_model(X_valid, f1=metric_f1)
    print(result)

    y_pred, raw_outputs = model.predict(test['text'])
    print(y_pred)

    test = pd.read_csv(DATASET_DIR+"test.csv")
    submit = pd.DataFrame({'index':test['id'], 'pred':y_pred+1})

    current_time = time.strftime('%Y%m%d-%H%M%S')
    filename = f'{current_time}_bert.csv'
    filepath = os.path.join(SUBMITS_DIR, filename)
    submit.to_csv(filepath, index=False, header=False)

    print(f'Saved {filename}')