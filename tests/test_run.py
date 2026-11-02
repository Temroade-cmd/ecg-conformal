import json
import subprocess
import sys

import numpy as np


def test_train_command(fake_cache, tmp_path):
    # la commande complète, comme elle sera lancée sur GitHub Actions
    subprocess.run([sys.executable, "-m", "ecgc.run", "train", "--cache", str(fake_cache), "--out", str(tmp_path),
                    "--epochs", "1", "--batch", "32"], check=True)
    assert (tmp_path / "model.pt").exists()
    assert len(json.loads((tmp_path / "history.json").read_text())) == 1
    # les probas de calibration, entre 0 et 1, une colonne par diagnostic
    cal = np.load(tmp_path / "calibration.npz")
    assert cal["P"].shape == cal["Y"].shape
    assert cal["P"].shape[1] == 5
    assert ((cal["P"] >= 0) & (cal["P"] <= 1)).all()
