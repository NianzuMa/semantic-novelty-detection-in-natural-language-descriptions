from sklearn.metrics import roc_auc_score


def get_auc_score(y_true, pred_score):
    auc_score = roc_auc_score(y_true, pred_score)
    return auc_score
