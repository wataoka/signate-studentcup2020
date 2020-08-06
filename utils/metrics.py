import numpy as np
from sklearn.metrics import f1_score

# f1 score for lgb
def metric_f1(preds, data):
    y_true = data.get_label()
    preds = preds.reshape(4, len(preds) // 4)
    y_pred = np.argmax(preds, axis=0)
    score = f1_score(y_true, y_pred, average="macro")
    return "metric_f1", score, True