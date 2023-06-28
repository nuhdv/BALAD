from sklearn import metrics

def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    return auroc, ap
