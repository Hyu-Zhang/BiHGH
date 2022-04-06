# """The :mod:`utils.metrics` module includes performance metrics."""
# import numpy as np
from sklearn import metrics
import numpy as np


_POSILABEL = 1
_NEGALABEL = 0


def MRR_NDCG_score(posi, nega):
    all_sum_mrr = 0
    all_sum_ndcg = 0
    all_num = len(posi)
    for p, n in zip(posi, nega):
        sum_ndcg = 0
        sum_mrr = 0
        num = len(p)
        flag = 0
        for pp in p:
            negi = n[flag:flag + 9]
            flag += 9

            posi_, nega_ = np.array(pp), np.array(negi)
            y_label = np.array([1] * 1 + [0] * len(nega_))

            y_score = np.hstack((posi_.flatten(), nega_.flatten()))

            order = np.argsort(-y_score)
            p_label = np.take(y_label, order)
            i_label = np.sort(y_label)[::-1]

            p_gain = 2 ** p_label - 1
            i_gain = 2 ** i_label - 1
            new_ndcg = 1 / (np.log2(list(p_gain).index(1) + 2))

            discounts = np.log2(np.maximum(np.arange(len(y_label)) + 1, 2.0))

            dcg_score = (p_gain / discounts).cumsum()
            idcg_score = (i_gain / discounts).cumsum()

            try:
                MRR_score = 1 / (list(p_gain).index(1) + 1)
            except:
                MRR_score = 0
            ndcg_score_ = (dcg_score / idcg_score).mean()
            # sum_ndcg += ndcg_score_
            sum_ndcg += new_ndcg
            sum_mrr += MRR_score
        mean_ndcg_ = sum_ndcg / num
        mean_mrr_ = sum_mrr / num
        all_sum_mrr += mean_mrr_
        all_sum_ndcg += mean_ndcg_
    return all_sum_ndcg / all_num, all_sum_mrr / all_num

def NDCG(posi, nega, wtype="max"):
    assert len(posi) == len(nega)
    u_labels, u_scores = [], []
    for p, n in zip(posi, nega):
        label, score = _canonical(p, n)
        u_labels.append(label)
        u_scores.append(score)
    return mean_ndcg_score(u_scores, u_labels, wtype)


def mean_ndcg_score(u_scores, u_labels, wtype="max"):
    num_users = len(u_scores)
    n_samples = [len(scores) for scores in u_scores]
    max_sample = max(n_samples)
    count = np.zeros(max_sample)
    mean_ndcg = np.zeros(num_users)
    avg_ndcg = np.zeros(max_sample)
    sum1 = 0
    for u in range(num_users):
        ndcg, mrr = ndcg_score(u_scores[u], u_labels[u], wtype)
        avg_ndcg[: n_samples[u]] += ndcg
        count[: n_samples[u]] += 1
        mean_ndcg[u] = ndcg.mean()
        sum1 += mrr
    return mean_ndcg


def ndcg_score(y_score, y_label, wtype="max"):
    order = np.argsort(-y_score)
    p_label = np.take(y_label, order)
    i_label = np.sort(y_label)[::-1]
    p_gain = 2 ** p_label - 1
    i_gain = 2 ** i_label - 1
    if wtype.lower() == "max":
        discounts = np.log2(np.maximum(np.arange(len(y_label)) + 1, 2.0))
    else:
        discounts = np.log2(np.arange(len(y_label)) + 2)
    dcg_score = (p_gain / discounts).cumsum()
    idcg_score = (i_gain / discounts).cumsum()
    MRR_score = 1 / (list(p_gain).index(1) + 1)

    return dcg_score / idcg_score, MRR_score


def _canonical(posi, nega):
    posi, nega = np.array(posi), np.array(nega)
    y_true = np.array([_POSILABEL] * len(posi) + [_NEGALABEL] * len(nega))
    y_score = np.hstack((posi.flatten(), nega.flatten()))
    return (y_true, y_score)


# ··············································#

def ROC(posi, nega):
    """Compute ROC and AUC for all user.

    Parameters
    ----------
    posi: Scores for positive outfits for each user.
    nega: Socres for negative outfits for each user.

    Returns
    -------
    roc: A tuple of (fpr, tpr, thresholds), see sklearn.metrics.roc_curve
    auc: AUC score.

    """
    assert len(posi) == len(nega)
    num = len(posi)
    mean_auc = 0.0
    aucs = []
    for p, n in zip(posi, nega):
        y_true, y_score = _canonical(p, n)
        auc = metrics.roc_auc_score(y_true, y_score)
        aucs.append(auc)
        mean_auc += auc
    mean_auc /= num
    return (aucs, mean_auc)
