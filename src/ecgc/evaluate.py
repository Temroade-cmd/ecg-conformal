"""Évaluation sur le fold de test (fold 10), une seule fois, avec le modèle final.

Trois mesures.
1. La macro AUROC, avec un intervalle de confiance à 95% calculé par bootstrap.
2. La prédiction conforme par diagnostic, pour vérifier que la garantie tient vraiment sur de nouveaux patients.
3. Le contrôle du risque, la part de diagnostics oubliés avec le seuil choisi sur la calibration.
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


def crc_report(P, Y, lam):
    # le seuil a été choisi sur la calibration, ici on regarde ce qu'il donne sur le test
    keep = Y.sum(1) > 0  # comme pour la calibration, un patient sans diagnostic ne peut rien oublier
    P, Y = P[keep], Y[keep]
    pred = P >= 1.0 - lam
    return {
        "lambda": float(lam),
        "threshold": float(1.0 - lam),
        "fnr": float(cp.fnr_loss(P, Y, lam).mean()),
        "mean_set_size": float(pred.sum(1).mean()),  # nombre moyen de diagnostics annoncés par patient
        "mean_true_labels": float(Y.sum(1).mean()),  # nombre moyen de vrais diagnostics par patient
    }


def evaluate(P_test, Y_test, P_cal, Y_cal, classes, alphas=(0.1, 0.05)):
    # la calibration n'utilise que le fold 9, tous les chiffres sont mesurés sur le fold 10
    cal = cp.calibration_scores(P_cal, Y_cal)
    auc, lo, hi = bootstrap_macro_auc(Y_test, P_test)
    res = {
        "n_test": int(len(Y_test)),
        "n_calibration": int(len(Y_cal)),
        "macro_auroc": {"value": auc, "ci95": [lo, hi]},
        "auroc_per_class": {c: float(roc_auc_score(Y_test[:, k], P_test[:, k])) for k, c in enumerate(classes)},
        "mondrian": {str(a): mondrian_report(P_test, Y_test, cal, a, classes) for a in alphas},
        "crc": {},
    }
    for a in alphas:
        lam = cp.crc_lambda(P_cal, Y_cal, a)
        res["crc"][str(a)] = crc_report(P_test, Y_test, lam)
    return res
