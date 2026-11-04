"""Évaluation sur le fold de test (fold 10), une seule fois, avec le modèle final.

Pour l'instant, la macro AUROC avec un intervalle de confiance à 95% calculé par bootstrap.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from . import conformal as cp


def bootstrap_macro_auc(Y, P, n_boot=1000, seed=0):
    # on refait le test 1000 fois en tirant les patients au hasard
    # replace=False pour ne jamais prendre deux fois le même patient
    rng = np.random.default_rng(seed)
    point = np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])])
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(Y), len(Y), replace=False)
        Yb, Pb = Y[idx], P[idx]
        if (Yb.sum(0) == 0).any() or (Yb.sum(0) == len(Yb)).any():
            continue  # tirage inutilisable si un diagnostic n'a que des 0 ou que des 1
        vals.append(np.mean([roc_auc_score(Yb[:, k], Pb[:, k]) for k in range(Y.shape[1])]))
    # l'intervalle à 95% va du 2,5e au 97,5e percentile des 1000 scores
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)
