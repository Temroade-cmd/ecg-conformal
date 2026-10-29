"""Entraînement du modèle."""

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(Y, P):
    # moyenne des AUROC des 5 classes (on saute une classe si elle n'a qu'une seule valeur)
    return float(np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in range(Y.shape[1]) if 0 < Y[:, k].sum() < len(Y)]))
