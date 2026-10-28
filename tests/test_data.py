import subprocess
import sys
from pathlib import Path

import numpy as np

from ecgc import data as D

ROOT = Path(__file__).resolve().parents[1]


def test_build_and_load_cache(tmp_path):
    # faux PTB-XL miniature, puis lecture comme on le fera avec le vrai
    fixture = tmp_path / "fixture"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_fixture.py"), "--out", str(fixture), "--n", "120"], check=True)
    D.build_cache(fixture, tmp_path / "cache.npz")
    ds = D.load_cache(tmp_path / "cache.npz")

    assert ds.X.shape[1:] == (12, 1000)  # 12 dérivations, 10 s à 100 Hz
    assert ds.Y.shape[1] == 5  # les 5 superclasses
    assert (ds.Y.sum(1) > 0).all()  # plus aucun ECG sans label
    assert set(np.unique(ds.folds)) == set(range(1, 11))


def test_split_keeps_folds_apart(tmp_path):
    fixture = tmp_path / "fixture"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_fixture.py"), "--out", str(fixture), "--n", "120"], check=True)
    D.build_cache(fixture, tmp_path / "cache.npz")
    ds = D.load_cache(tmp_path / "cache.npz")

    train, test = ds.split(D.TRAIN_FOLDS), ds.split(D.TEST_FOLD)
    assert set(train.ecg_ids).isdisjoint(test.ecg_ids)  # aucun ECG à la fois en train et en test


def test_superclass_mapping():
    import pandas as pd

    statements = pd.DataFrame(
        {"diagnostic": [1.0, 1.0, None], "diagnostic_class": ["MI", "HYP", None]}, index=["IMI", "LVH", "SR"]
    )
    meta = pd.DataFrame({"scp_codes": ["{'IMI': 100.0, 'SR': 0.0}", "{'LVH': 50.0, 'IMI': 15.0}"]})
    Y = D.superdiagnostic_labels(meta, statements)
    # SR n'est pas un diagnostic, il est ignoré. IMI compte même avec un score bas.
    assert Y.tolist() == [[0, 1, 0, 0, 0], [0, 1, 0, 0, 1]]
