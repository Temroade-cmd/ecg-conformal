import numpy as np

from ecgc import conformal as cp


def simulate(rng, n, k=5):
    # fausses probas d'un classifieur moyen, pour vérifier les garanties sans entraîner de modèle
    Y = (rng.random((n, k)) < 0.3).astype(float)
    logits = 1.5 * (2 * Y - 1) + rng.standard_normal((n, k))
    return 1 / (1 + np.exp(-logits)), Y


def test_pvalue_formula():
    cal = np.sort(np.array([0.1, 0.2, 0.3, 0.4]))
    p = cp._pvalues(np.array([0.25, 0.0, 0.5]), cal)
    # 0.25, 2 scores sur 4 sont >= 0.25, donc (1 + 2) / 5
    # 0.5, plus bizarre que tout le monde, mais la p-valeur ne tombe jamais à 0
    np.testing.assert_allclose(p, [(1 + 2) / 5, (1 + 4) / 5, 1 / 5])


def test_mondrian_coverage_per_label():
    rng = np.random.default_rng(0)
    alpha, covs_pos, covs_neg = 0.1, [], []
    for _ in range(200):
        P_cal, Y_cal = simulate(rng, 400)
        P_te, Y_te = simulate(rng, 400)
        inc_pos, inc_neg = cp.mondrian_sets(P_te, cp.calibration_scores(P_cal, Y_cal), alpha)
        covs_pos.append(inc_pos[Y_te == 1].mean())
        covs_neg.append(inc_neg[Y_te == 0].mean())
    # la garantie est vraie en moyenne sur beaucoup de tirages de calibration
    assert np.mean(covs_pos) >= 1 - alpha - 0.005
    assert np.mean(covs_neg) >= 1 - alpha - 0.005


def test_states():
    inc_pos = np.array([[True, False, True, False]])
    inc_neg = np.array([[False, True, True, False]])
    assert list(cp.states(inc_pos, inc_neg)[0]) == ["present", "absent", "uncertain", "atypical"]
