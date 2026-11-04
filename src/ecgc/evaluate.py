"""Évaluation sur le fold de test (fold 10), une seule fois, avec le modèle final.

1. La macro AUROC, avec un intervalle de confiance à 95% calculé par bootstrap.
2. La prédiction conforme par diagnostic, pour vérifier que la garantie tient vraiment sur de nouveaux patients.
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


def mondrian_report(P, Y, cal, alpha, classes):
    # pour chaque diagnostic, ce que la prédiction conforme donne vraiment sur le test
    inc_pos, inc_neg = cp.mondrian_sets(P, cal, alpha)
    st = cp.states(inc_pos, inc_neg)
    per = {}
    for k, name in enumerate(classes):
        y = Y[:, k] == 1
        decided = (st[:, k] == "present") | (st[:, k] == "absent")
        correct = ((st[:, k] == "present") & y) | ((st[:, k] == "absent") & ~y)
        per[name] = {
            # la garantie, au moins 1 - alpha chez les malades et chez les non malades
            "coverage_pos": float(inc_pos[y, k].mean()) if y.any() else None,
            "coverage_neg": float(inc_neg[~y, k].mean()) if (~y).any() else None,
            # le prix de la garantie, la part de patients pour qui le modèle tranche
            "decided_rate": float(decided.mean()),
            "accuracy_when_decided": float(correct[decided].mean()) if decided.any() else None,
            "uncertain_rate": float((st[:, k] == "uncertain").mean()),
            "atypical_rate": float((st[:, k] == "atypical").mean()),
        }
    return per
