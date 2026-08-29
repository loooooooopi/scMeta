"""Fast DeLong test for comparing two correlated ROC AUCs (same y_true, two
different score vectors from models evaluated on the identical cells) --
Sun & Xu (2014) O(N log N) implementation. Used to test whether
scMeta-graphloss's AUC improvement over MLP / original scMeta within a given
CV/LOCO split is distinguishable from evaluation noise, since fold-level
paired t-tests across only 5 folds are underpowered.
"""
import numpy as np
from scipy import stats


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Returns (auc_a, auc_b, z, p) for two models' scores on the same y_true."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="stable")
    y_true_sorted = y_true[order]
    preds_sorted = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    label_1_count = int(y_true_sorted.sum())
    aucs, delongcov = _fast_delong(preds_sorted, label_1_count)
    l = np.array([[1, -1]])
    var = float(np.dot(np.dot(l, delongcov), l.T)[0, 0])
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return aucs[0], aucs[1], z, p


if __name__ == "__main__":
    # self-test 1: identical scores -> AUC identical, p == 1
    rng = np.random.default_rng(0)
    y = (rng.random(2000) > 0.5).astype(int)
    p1 = rng.random(2000) + y * 0.5
    a, b, z, p = delong_roc_test(y, p1, p1)
    assert abs(a - b) < 1e-12 and abs(z) < 1e-9 and abs(p - 1.0) < 1e-9, (a, b, z, p)

    # self-test 2: cross-check AUC value against sklearn
    from sklearn.metrics import roc_auc_score
    p2 = rng.random(2000) + y * 0.2
    a, b, z, p = delong_roc_test(y, p1, p2)
    assert abs(a - roc_auc_score(y, p1)) < 1e-9
    assert abs(b - roc_auc_score(y, p2)) < 1e-9

    # self-test 3: strongly different score sets should give a very small p-value
    p3 = rng.random(2000)  # near-random scores, should differ a lot from p1 (AUC ~0.5 vs ~0.7+)
    a, b, z, p = delong_roc_test(y, p1, p3)
    print(f"self-test 3: auc_a={a:.4f} auc_b={b:.4f} z={z:.3f} p={p:.2e}")
    assert p < 0.01

    print("delong.py self-tests passed.")
