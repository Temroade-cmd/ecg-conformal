import numpy as np

from ecgc import evaluate as ev
from ecgc import conformal as cp

CLASSES = ["A", "B", "C", "D", "E"]


def simulate(rng, n, k=5):
    # même faux classifieur que dans test_conformal.py
    Y = (rng.random((n, k)) < 0.3).astype(float)
    logits = 1.5 * (2 * Y - 1) + rng.standard_normal((n, k))
    return 1 / (1 + np.exp(-logits)), Y


def test_bootstrap_gives_a_real_interval():
    P, Y = simulate(np.random.default_rng(0), 500)
    point, lo, hi = ev.bootstrap_macro_auc(Y, P, n_boot=200)
    # le score mesuré est dans l'intervalle, et l'intervalle n'est pas réduit à un point
    assert lo <= point <= hi
    assert hi - lo > 0.005


def test_mondrian_report_on_new_patients():
    rng = np.random.default_rng(1)
    P_cal, Y_cal = simulate(rng, 2000)
    P_te, Y_te = simulate(rng, 2000)
    per = ev.mondrian_report(P_te, Y_te, cp.calibration_scores(P_cal, Y_cal), 0.1, CLASSES)
    for r in per.values():
        # avec 2000 patients de calibration, la couverture est proche de 90%
        assert r["coverage_pos"] > 0.86 and r["coverage_neg"] > 0.86
        # chaque patient est dans un seul des trois cas
        assert abs(r["decided_rate"] + r["uncertain_rate"] + r["atypical_rate"] - 1) < 1e-9


def test_crc_report_counts_only_patients_with_a_diagnosis():
    P = np.full((4, 5), 0.9)
    Y = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float)
    # avec lam = 1 tout est annoncé, donc aucun vrai diagnostic n'est oublié
    r = ev.crc_report(P, Y, 1.0)
    assert r["fnr"] == 0.0
    assert r["mean_true_labels"] == 4 / 3


def test_evaluate_gives_every_number():
    rng = np.random.default_rng(2)
    P_cal, Y_cal = simulate(rng, 300)
    P_te, Y_te = simulate(rng, 300)
    res = ev.evaluate(P_te, Y_te, P_cal, Y_cal, CLASSES)
    assert set(res) == {"n_test", "n_calibration", "macro_auroc", "auroc_per_class", "mondrian", "crc"}
    assert set(res["mondrian"]) == {"0.1", "0.05"}
    assert res["crc"]["0.1"]["fnr"] <= 0.2
