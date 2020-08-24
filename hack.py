import os
import time

import pulp  # pip install pulp==2.3
import numpy as np
import pandas as pd

from constants import OUTPUTS_DIR, DATASET_DIR, SUBMITS_DIR

N_CLASSES = [404, 320, 345, 674]  # @yCarbonによる推定（過去フォーラム参照）

# 制約付き対数尤度最大化問題を解く
def hack(prob):
    logp = np.log(prob + 1e-16)
    N = prob.shape[0]
    K = prob.shape[1]

    m = pulp.LpProblem('Problem', pulp.LpMaximize)  # 最大化問題

    # 最適化する変数(= 提出ラベル)
    xx = pulp.LpVariable.dicts('xx', [(i, j) for i in range(N) for j in range(K)], 0, 1, pulp.LpBinary)
    
    # log likelihood(目的関数)
    log_likelihood = pulp.lpSum([xx[(i, j)] * logp[i, j] for i in range(N) for j in range(K)])
    m += log_likelihood
    
    # 各データについて，1クラスだけを予測ラベルとする制約
    for i in range(N):
        m += pulp.lpSum([xx[(i, k)] for k in range(K)]) == 1  # i.e., SOS1
    
    # 各クラスについて，推定個数の合計に関する制約
    for k in range(K):
        m += pulp.lpSum([xx[(i, k)] for i in range(N)]) == N_CLASSES[k]
        
    m.solve()  # 解く

    assert m.status == 1  # assert 最適 <=>（実行可能解が見つからないとエラー）

    x_ast = np.array([[int(xx[(i, j)].value()) for j in range(K)] for i in range(N)])  # 結果の取得
    return x_ast.argmax(axis=1)

filepath = os.path.join(OUTPUTS_DIR, 'bert_outputs.npy')
raw_outputs = np.load(filepath)
norm_outputs = (raw_outputs-np.min(raw_outputs))/(np.max(raw_outputs)-np.min(raw_outputs))

y_pred = hack(norm_outputs)

filepath = os.path.join(DATASET_DIR, 'test.csv')
test = pd.read_csv(filepath)
submit = pd.DataFrame({'index':test['id'], 'pred':y_pred+1})

current_time = time.strftime('%Y%m%d-%H%M%S')
filename = f'{current_time}_bert.csv'
filepath = os.path.join(SUBMITS_DIR, filename)
submit.to_csv(filepath, index=False, header=False)

print(f'Saved {filename}')