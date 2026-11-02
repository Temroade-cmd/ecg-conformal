"""Prédiction conforme appliquée aux sorties du classifieur.

Le modèle donne une proba par diagnostic. La prédiction conforme compare cette proba
à celles de patients dont on connaît déjà la réponse (la calibration, fold 9),
et en tire une réponse qui a une garantie.

Conforme par label (Mondrian). Pour chaque diagnostic, on regarde si la proba du patient
ressemble à celles des vrais positifs, des vrais négatifs, des deux, ou d'aucun.
Réponse possible, présent, absent, je ne sais pas, ou atypique.
"""

import numpy as np


# conforme par label

def calibration_scores(probs, Y):
    # un score de "bizarrerie", plus il est grand moins le patient ressemble à son groupe
    # vrai positif, score = 1 - p (un positif avec une proba basse est bizarre)
    # vrai négatif, score = p (un négatif avec une proba haute est bizarre)
    # les listes sont triées une fois pour toutes, le comptage sera plus rapide après
    out = []
    for k in range(probs.shape[1]):
        pos = np.sort(1.0 - probs[Y[:, k] == 1, k])
        neg = np.sort(probs[Y[:, k] == 0, k])
        out.append({"pos": pos, "neg": neg})
    return out


def _pvalues(scores, cal_sorted):
    # p-valeur conforme = part des scores de calibration >= score
    # la liste est triée donc searchsorted fait le comptage d'un coup
    n = len(cal_sorted)
    n_ge = n - np.searchsorted(cal_sorted, scores, side="left")
    return n_ge / n


def pvalues(probs, cal):
    # deux tableaux de la même forme que probs, p-valeur "présent" et p-valeur "absent"
    p_pos = np.empty_like(probs, dtype=np.float64)
    p_neg = np.empty_like(probs, dtype=np.float64)
    for k, c in enumerate(cal):
        p_pos[:, k] = _pvalues(1.0 - probs[:, k], c["pos"])
        p_neg[:, k] = _pvalues(probs[:, k], c["neg"])
    return p_pos, p_neg
