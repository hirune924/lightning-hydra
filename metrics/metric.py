import sklearn.metrics as metrics
import numpy as np


def lazy_accuracy(y, preds, num_classes=6, verbose=False):
    cm = metrics.confusion_matrix(y, preds, labels=range(num_classes))
    weight = (
        np.eye(num_classes, k=0) + np.eye(num_classes, k=1) + np.eye(num_classes, k=-1)
    )
    if verbose:
        print(cm)
    return (cm * weight).sum() / cm.sum()
