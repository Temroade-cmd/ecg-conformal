"""Prédiction conforme appliquée aux sorties du classifieur.

Le modèle donne une proba par diagnostic. La prédiction conforme compare cette proba
à celles de patients dont on connaît déjà la réponse (la calibration, fold 9),
et en tire une réponse qui a une garantie.

Deux méthodes ici.
1. Conforme par label (Mondrian). Pour chaque diagnostic, on regarde si la proba du patient
   ressemble à celles des vrais positifs, des vrais négatifs, des deux, ou d'aucun.
   Réponse possible, présent, absent, je ne sais pas, ou atypique.
2. Contrôle du risque. Un seul seuil qui limite le taux de diagnostics oubliés.
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
    # p-valeur conforme = (1 + nb de scores de calibration >= score) / (n + 1)
    # le +1 compte le nouveau patient lui-même, il fait partie des n + 1
    # la liste est triée donc searchsorted fait le comptage d'un coup
    n = len(cal_sorted)
    n_ge = n - np.searchsorted(cal_sorted, scores, side="left")
    return (1.0 + n_ge) / (n + 1.0)


def pvalues(probs, cal):
    # deux tableaux de la même forme que probs, p-valeur "présent" et p-valeur "absent"
    p_pos = np.empty_like(probs, dtype=np.float64)
    p_neg = np.empty_like(probs, dtype=np.float64)
    for k, c in enumerate(cal):
        p_pos[:, k] = _pvalues(1.0 - probs[:, k], c["pos"])
        p_neg[:, k] = _pvalues(probs[:, k], c["neg"])
    return p_pos, p_neg


def mondrian_sets(probs, cal, alpha):
    # on garde une réponse si sa p-valeur dépasse alpha
    # avec alpha = 0.1, un vrai positif garde "présent" au moins 90% du temps, pareil pour les négatifs
    p_pos, p_neg = pvalues(probs, cal)
    return p_pos > alpha, p_neg > alpha


def states(inc_pos, inc_neg):
    # une seule réponse gardée = le modèle tranche, les deux = il ne sait pas, aucune = cas atypique
    s = np.full(inc_pos.shape, "atypical", dtype=object)
    s[inc_pos & ~inc_neg] = "present"
    s[~inc_pos & inc_neg] = "absent"
    s[inc_pos & inc_neg] = "uncertain"
    return s


# contrôle du risque, taux de diagnostics oubliés

def fnr_loss(probs, Y, lam):
    # on annonce tous les diagnostics qui ont une proba >= 1 - lam
    # pour chaque patient, la part de ses vrais diagnostics qu'on a ratés
    pred = probs >= 1.0 - lam
    hits = (pred & (Y == 1)).sum(1)
    return 1.0 - hits / np.maximum(Y.sum(1), 1)


def crc_lambda(probs, Y, alpha, grid=2001):
    # plus lam est grand, plus on annonce de diagnostics, et moins on en oublie
    # on cherche le plus petit lam qui respecte n/(n+1) * risque + 1/(n+1) <= alpha
    keep = Y.sum(1) > 0  # un patient sans aucun diagnostic ne peut rien oublier
    probs, Y = probs[keep], Y[keep]
    n = len(Y)
    for lam in np.linspace(0.0, 1.0, grid):
        risk = fnr_loss(probs, Y, lam).mean()
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return float(lam)
    return 1.0
