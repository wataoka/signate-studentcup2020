import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from constants import BASE_DIR
from modules import JobInfoDataset, NNTfidf
from utils.data import load_dataset, get_weight
from utils.features import word2vec
from utils.utils import fix_seed
from utils.observer import get_current_time

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

    n_folds = 5
    epochs = 200
    batch_size = 512

    # data
    train_df, test_df, sample_submit_df = load_dataset()
    X, X_test = word2vec(train_df, test_df)

    X = X.values.astype('float32')
    X_test = X_test.values.astype('float32')
    y = pd.get_dummies(train_df['jobflag']).values.astype('float32')

    trainset = JobInfoDataset(X, y, jobflag=train_df['jobflag'].values)
    testset = JobInfoDataset(X_test)

    # weight
    weight = get_weight(train_df)

    # ---------- Kfold ---------- #
    preds_for_test = [[0 for _ in range(4)] for _ in range(len(X_test))]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=seed)
    cv_loss_list = []
    cv_acc_list = []
    cv_f1_list = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(trainset.X, trainset.jobflag)):
        print(f'\nFold {fold_idx+1}')

        # model
        model = NNTfidf()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):

            # dataloader
            sampler = WeightedRandomSampler(weight[train_idx], len(weight[train_idx]))
            trainloader = DataLoader(Subset(trainset, train_idx),
                                     sampler=sampler,
                                     batch_size=batch_size)
            validloader = DataLoader(Subset(trainset, valid_idx),
                                     batch_size=batch_size)

            # train
            model.train()
            running_loss = []
            num_correct = 0
            num_total = 0

            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, pred = torch.max(outputs.data, 1)
                _, true = torch.max(labels.data, 1)
                num_correct += (pred == true).sum().item()
                num_total += len(labels)
                running_loss.append(loss.item())

            train_loss = np.mean(running_loss)
            train_acc = num_correct/num_total

            # valid
            model.eval()
            running_loss = []
            num_correct = 0
            num_total = 0
            pred_list = []
            true_list = []

            with torch.no_grad():
                for inputs, labels in validloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss.append(loss)

                    _, pred = torch.max(outputs.data, 1)
                    _, true = torch.max(labels.data, 1)
                    num_correct += (pred == true).sum().item()
                    num_total += len(labels)

                    true_list += list(true.numpy())
                    pred_list += list(pred.numpy())

            valid_loss = np.mean(running_loss)
            valid_acc = num_correct/num_total

            valid_f1 = f1_score(true_list, pred_list, average="macro")

            if (epoch+1)%100 == 0 or epoch == 0:
                print(f'Epoch:{epoch+1}/{epochs} \t '    \
                      f'train_loss: {train_loss:.3f},  ' \
                      f'train_acc: {train_acc:.3f},  '   \
                      f'valid_loss: {valid_loss:.3f},  ' \
                      f'valid_acc: {valid_acc:.3f},  '   \
                      f'valid F1: {valid_f1:.3f}')
        
        cv_loss_list.append(valid_loss)
        cv_acc_list.append(valid_acc)
        cv_f1_list.append(valid_f1)

        # pred for test
        testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)
        with torch.no_grad():
            for inputs in testloader:
                outputs = model(inputs)
                preds_for_test += outputs.numpy()/n_folds

    cv_loss = np.mean(cv_loss_list)
    cv_acc = np.mean(cv_acc_list)
    cv_f1 = np.mean(cv_f1_list)
    print(f'\n' \
          f'cv valid loss: {cv_loss:.3f},  ' \
          f'cv valid acc: {cv_acc:.3f},  '   \
          f'cv valid f1: {cv_f1:.3f},  ')
    
    if args.submit:
        pred = np.argmax(preds_for_test, axis=1)+1
        submit = pd.DataFrame({'index':test_df['id'], 'pred':pred})

        model_name = "nn_word2vec"
        current_time = get_current_time()
        filename = f"{current_time}_{model_name}.csv"
        filepath = os.path.join('submits', filename)
        submit.to_csv(filepath, index=False, header=False)

        print(f'Save {filepath}')