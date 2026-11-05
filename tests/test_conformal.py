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


def test_fnr_goes_down_when_lambda_goes_up():
    rng = np.random.default_rng(1)
    P, Y = simulate(rng, 300)
    keep = Y.sum(1) > 0  # seulement les patients qui ont au moins un diagnostic
    P, Y = P[keep], Y[keep]
    fnr = [cp.fnr_loss(P, Y, lam).mean() for lam in (0.0, 0.3, 0.6, 0.9, 1.0)]
    # lam = 0, on n'annonce rien, tout est oublié. lam = 1, on annonce tout, rien n'est oublié
    assert all(a >= b for a, b in zip(fnr, fnr[1:]))
    assert fnr[-1] == 0.0


def test_risk_control_limits_missed_diagnoses():
    rng = np.random.default_rng(2)
    alpha, fnrs = 0.1, []
    for _ in range(200):
        P_cal, Y_cal = simulate(rng, 300)
        P_te, Y_te = simulate(rng, 300)
        lam = cp.crc_lambda(P_cal, Y_cal, alpha)
        keep = Y_te.sum(1) > 0
        fnrs.append(cp.fnr_loss(P_te[keep], Y_te[keep], lam).mean())
    # en moyenne sur les tirages, on oublie au plus 10% des vrais diagnostics
    assert np.mean(fnrs) <= alpha + 0.005


def test_decision_bounds_match_pvalue_rule():
    # les seuils a et b doivent donner exactement la même décision que les p-valeurs
    rng = np.random.default_rng(3)
    P_cal, Y_cal = simulate(rng, 300)
    cal = cp.calibration_scores(P_cal, Y_cal)
    grid = np.linspace(0.0005, 0.9995, 1000)[:, None].repeat(5, 1)
    for alpha in (0.01, 0.05, 0.1, 0.2, 0.5):
        inc_pos, inc_neg = cp.mondrian_sets(grid, cal, alpha)
        for k, c in enumerate(cal):
            a, b = cp.decision_bounds(c["pos"], c["neg"], alpha)
            assert np.array_equal(inc_pos[:, k], grid[:, k] >= a)
            assert np.array_equal(inc_neg[:, k], grid[:, k] <= b)
