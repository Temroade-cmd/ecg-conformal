import json
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

from ecgc import data as D
from ecgc.export import export_onnx
from ecgc.model import Deployable, ECGResNet


def test_onnx_gives_the_same_probabilities(tmp_path):
    # l'app doit donner exactement les mêmes probas que le modèle PyTorch
    torch.manual_seed(0)
    net = ECGResNet()
    rng = np.random.default_rng(0)
    mean = rng.normal(0, 0.1, 12).astype(np.float32)
    std = rng.uniform(0.5, 1.5, 12).astype(np.float32)
    export_onnx(net, mean, std, tmp_path / "model.onnx")

    x = rng.normal(0, 1, (3, 12, 1000)).astype(np.float32)  # des "millivolts", comme dans l'app
    with torch.no_grad():
        want = Deployable(net, mean, std).eval()(torch.as_tensor(x)).numpy()
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(["prob"], {"ecg": x})[0]
    np.testing.assert_allclose(got, want, atol=1e-4)
    assert ((got >= 0) & (got <= 1)).all()


def test_export_command(fake_cache, tmp_path):
    # la commande complète, comme dans le workflow de publication
    run = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "ecgc.run", "train", "--cache", str(fake_cache), "--out", str(run),
                    "--epochs", "1", "--batch", "32"], check=True)
    subprocess.run([sys.executable, "-m", "ecgc.run", "export", "--run", str(run), "--cache", str(fake_cache),
                    "--out", str(tmp_path / "bundle")], check=True)
    meta = json.loads((tmp_path / "bundle" / "meta.json").read_text())
    samples = json.loads((tmp_path / "bundle" / "samples.json").read_text())
    assert (tmp_path / "bundle" / "ecg_resnet.onnx").exists()
    assert meta["classes"] == D.CLASSES
    # chaque patient de calibration est soit positif soit négatif pour chaque diagnostic
    n_cal = len(D.load_cache(fake_cache).split(D.CAL_FOLD))
    assert all(len(c["pos"]) + len(c["neg"]) == n_cal for c in meta["calibration"].values())
    assert samples and np.asarray(samples[0]["signal"]).shape == (12, 1000)
