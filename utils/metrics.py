import pulp
import numpy as np
from sklearn.metrics import f1_score

# f1 score for lgb
def metric_f1(preds, data):
    y_true = data.get_label()
    preds = preds.reshape(4, len(preds) // 4)
    y_pred = np.argmax(preds, axis=0)
    score = f1_score(y_true, y_pred, average="macro")
    return "metric_f1", score, True

# 制約付き対数尤度最大化問題を解く
def hack(prob):

    N_CLASSES = [404, 320, 345, 674]  # @yCarbonによる推定（過去フォーラム参照）

    logp = np.log(prob + 1e-16)
    N = prob.shape[0]
    K = prob.shape[1]

    m = pulp.LpProblem('Problem', pulp.LpMaximize)  # 最大化問題

    # 最適化する変数(= 提出ラベル)
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(N) for j in range(K)], 0, 1, pulp.LpBinary)
    
    # log likelihood(目的関数)
    log_likelihood = pulp.lpSum([x[(i, j)] * logp[i, j] for i in range(N) for j in range(K)])
    m += log_likelihood
    
    # 各データについて，1クラスだけを予測ラベルとする制約
    for i in range(N):
        m += pulp.lpSum([x[(i, k)] for k in range(K)]) == 1  # i.e., SOS1
    
    # 各クラスについて，推定個数の合計に関する制約
    for k in range(K):
        m += pulp.lpSum([x[(i, k)] for i in range(N)]) == N_CLASSES[k]
        
    m.solve()  # 解く

    assert m.status == 1  # assert 最適 <=>（実行可能解が見つからないとエラー）

    x_ast = np.array([[int(x[(i, j)].value()) for j in range(K)] for i in range(N)])  # 結果の取得
    return x_ast.argmax(axis=1) # 結果をonehotから -> {0, 1, 2, 3}のラベルに変換