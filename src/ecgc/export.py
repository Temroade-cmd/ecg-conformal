"""Tout ce dont l'app a besoin. Le modèle en ONNX, les scores de calibration et quelques ECG du test.

L'app n'installe pas PyTorch (trop lourd pour un petit serveur), elle utilise onnxruntime.
"""

import json
from pathlib import Path

import numpy as np
import torch

from . import conformal as cp
from .data import CLASS_NAMES, CLASSES, FS, LEADS, N_SAMPLES


def export_onnx(net, mean, std, path):
    # ONNX est un format standard, le modèle peut ensuite tourner sans PyTorch
    # axe batch dynamique, pour pouvoir envoyer plusieurs ECG d'un coup
    model = net.eval()
    dummy = torch.zeros(1, len(LEADS), N_SAMPLES)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["ecg"],
        output_names=["prob"],
        dynamic_axes={"ecg": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def pick_samples(P, Y, cal, seed=0):
    # un mélange lisible pour la démo
    # des cas à un seul diagnostic, des cas à plusieurs diagnostics, et des cas où le modèle hésite
    rng = np.random.default_rng(seed)
    chosen = []
    single = Y.sum(1) == 1
    for k in range(Y.shape[1]):
        cand = np.where(single & (Y[:, k] == 1))[0]
        if len(cand):
            cand = cand[np.argsort(-P[cand, k])]
            chosen.append(int(cand[0]))  # le cas le plus net
            if len(cand) > 1:
                chosen.append(int(cand[len(cand) // 2]))  # un cas moyen
    multi = np.where(Y.sum(1) >= 2)[0]
    chosen += [int(i) for i in rng.choice(multi, size=min(3, len(multi)), replace=False)]
    inc_pos, inc_neg = cp.mondrian_sets(P, cal, 0.1)
    unsure = np.where((inc_pos & inc_neg).sum(1) >= 2)[0]  # au moins deux "je ne sais pas"
    chosen += [int(i) for i in rng.choice(unsure, size=min(3, len(unsure)), replace=False)]
    return list(dict.fromkeys(chosen))  # enlève les doublons en gardant l'ordre


def write_bundle(out_dir, net, mean, std, P_cal, Y_cal, P_test, Y_test, test, results):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(net, mean, std, out_dir / "ecg_resnet.onnx")

    cal = cp.calibration_scores(P_cal, Y_cal)
    meta = {
        "classes": CLASSES,
        "class_names": CLASS_NAMES,
        "leads": LEADS,
        "fs": FS,
        "n_samples": N_SAMPLES,
        # les scores de calibration, l'app recalcule les p-valeurs avec
        "calibration": {
            c: {"pos": np.round(s["pos"], 5).tolist(), "neg": np.round(s["neg"], 5).tolist()}
            for c, s in zip(CLASSES, cal)
        },
        "results": results,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    samples = [
        {
            "ecg_id": int(test.ecg_ids[i]),
            "labels": [c for k, c in enumerate(CLASSES) if Y_test[i, k] == 1],
            "signal": np.round(test.X[i].astype(np.float32), 3).tolist(),
        }
        for i in pick_samples(P_test, Y_test, cal)
    ]
    (out_dir / "samples.json").write_text(json.dumps(samples, separators=(",", ":")))
    return out_dir
