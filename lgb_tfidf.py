import os
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from utils.data import load_dataset, get_weight
from utils.features import tfidf
from utils.metrics import metric_f1
from utils.observer import get_current_time
from utils.utils import fix_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', type=str, default="False")
    args = parser.parse_args()
    args.submit = (args.submit == 'True')
    return args

args = parse_args()

if __name__ == "__main__":

    # hyper params
    seed=1; fix_seed(seed)

    max_features = 3000

    n_folds = 3
    num_boost_round = 1000
    early_stopping_rounds = 100

    # data
    train_df, test_df, sample_submit_df = load_dataset()
    X, X_test = tfidf(train_df, test_df, max_features=max_features)
    y = train_df['jobflag'].astype(int)-1

    weight = get_weight(train_df)

    # ---------- Kfold ---------- #
    scores = []
    y_test_pred = np.zeros((X_test.shape[0], y.nunique()), dtype='float32')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        train_data = lgb.Dataset(X_train, label=y_train, weight=weight[train_idx])
        valid_data = lgb.Dataset(X_valid, label=y_valid, weight=weight[valid_idx])

        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'None',
            'verbose': -1,
            'seed': seed
        }

        # train
        model = lgb.train(params,
                          train_data,
                          valid_sets=[train_data,valid_data],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=50,
                          feval=metric_f1)
        
        # evaluate
        y_val_pred = np.argmax(model.predict(X_valid), axis=1)
        score = f1_score(y_valid, y_val_pred, average='macro')
        scores.append(score)
        print(f"\nFold-{i+1}: Score: {score:.4f}\n")

        # predict test
        y_test_pred += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
    
    # evaluate
    print(f"Kfold F1 Score: {np.mean(scores):.4f}")
    
    # submit
    if args.submit:
        pred = np.argmax(y_test_pred, axis=1)+1
        submit = pd.DataFrame({'index':test_df['id'], 'pred':pred})

        model_name = "lgb_tfidf"
        current_time = get_current_time()
        filename = f"{current_time}_{model_name}.csv"
        filepath = os.path.join('submits', filename)
        submit.to_csv(filepath, index=False, header=False)

        print(f'Save {filepath}')