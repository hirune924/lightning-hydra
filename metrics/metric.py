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


def monitored_cohen_kappa_score(y1, y2, labels=None, weights=None,
                      sample_weight=None, verbose=False):
    confusion = metrics.confusion_matrix(y1, y2, labels=labels,
                                 sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    o = np.sum(w_mat * confusion)/(np.sum(confusion)*n_classes*n_classes)
    e = np.sum(w_mat * expected)/(np.sum(confusion)*n_classes*n_classes)
    k = o / e
    if verbose:
        print(confusion)
    return 1 - k, o, e