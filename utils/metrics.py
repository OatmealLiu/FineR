import torchmetrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def classification_acc(config: dict, top_k: int = 1):
    acc = torchmetrics.Accuracy(task="multiclass",
                                num_classes=config['num_classes'],
                                top_k=top_k).to(config['device'])
    return acc


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_arr, jnd_arr = linear_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_acc(predictions, labels):
    acc = cluster_acc(labels.astype(int), predictions.astype(int))
    nmi = nmi_score(labels, predictions)
    ari = ari_score(labels, predictions)
    return acc, nmi, ari

