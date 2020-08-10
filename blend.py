import os
import argparse
import pandas as pd

from constants import SUBMITS_DIR
from utils.data import load_dataset
from utils.observer import get_current_time

def add_onehot(sub):
    onehot = pd.get_dummies(sub['pred'])
    sub = pd.concat([sub, onehot], axis=1)
    return sub

def select_pred(d):

    preds = [d[1], d[2], d[3], d[4]]
    max_preds = max(preds)

    candidates = []
    for i in range(1, 5):
        if d[i] == max_preds:
            candidates.append(d[i])
    
    if candidates == []:
        print('candidates is empty!')
        exit(1)
    
    # adopt with priorites according to label distribution
    pred = None
    priorites = [3, 1, 4, 2]
    for i in priorites:
        if d[i] in candidates:
            pred = i
            break
    
    if pred is None:
        print('pred is None!')
        exit(1)
    
    return pred

if __name__ == "__main__":
    
    # config
    blend_list = [
        ['20200810-203854_lgb.csv', 0.4],
        ['20200810-213106_lgb_importance.csv', 0.3],
        ['submit_model2_bert.csv', 0.3]
    ]

    # prepare submit
    _, test_df, _ = load_dataset()
    submit = pd.DataFrame([])
    submit['id'] = test_df['id']
    submit[1] = 0
    submit[2] = 0
    submit[3] = 0
    submit[4] = 0

    # combine
    for filename, weight in blend_list:
        filepath = os.path.join(SUBMITS_DIR, filename)
        sub = pd.read_csv(filepath, names=('id', 'pred'))
        sub = add_onehot(sub)
        for i in range(1, 5):
            submit[i] += sub[i]*weight

    # postprocess
    submit['pred'] = submit.apply(select_pred, axis=1)
    submit = submit.drop([1, 2, 3, 4], axis=1)

    # save
    current_time = get_current_time()
    filename = f'{current_time}_blend.csv'
    filepath = os.path.join(SUBMITS_DIR, filename)
    submit.to_csv(filepath, index=False, header=False)