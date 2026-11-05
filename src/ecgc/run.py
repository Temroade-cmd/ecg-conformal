"""Lance le pipeline.

    python -m ecgc.run prepare --ptbxl chemin/vers/ptb-xl --cache data/ptbxl100.npz
    python -m ecgc.run train   --cache data/ptbxl100.npz --out runs/main
    python -m ecgc.run export  --run runs/main --cache data/ptbxl100.npz --out bundle

train entraîne le modèle, garde les probas de calibration, puis évalue une seule fois sur le test.
export prépare ce dont l'app a besoin à partir d'un run d'entraînement.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .evaluate import evaluate
from .export import write_bundle
from .model import ECGResNet
from .train import fit, predict


def cmd_prepare(a):
    print(f"Reading PTB-XL from {a.ptbxl}")
    D.build_cache(Path(a.ptbxl), Path(a.cache))
    print(f"Cache written to {a.cache}")


def cmd_train(a):
    torch.set_num_threads(a.threads or torch.get_num_threads())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = D.load_cache(Path(a.cache))
    # le test (fold 10) ne sert qu'une fois, tout à la fin, quand le modèle et la calibration sont figés
    train, val = ds.split(D.TRAIN_FOLDS), ds.split(D.VAL_FOLD)
    cal, test = ds.split(D.CAL_FOLD), ds.split(D.TEST_FOLD)
    print(f"train {len(train)}  val {len(val)}  calibration {len(cal)}  test {len(test)}")

    # normalisation calculée sur le train uniquement, pour ne rien apprendre du test
    mean, std = D.lead_stats(train.X)
    net, history = fit(train, val, mean, std, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed)

    # on sauve les poids et la normalisation ensemble, l'un ne sert à rien sans l'autre
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))

    # les probas des patients de calibration (fold 9), la prédiction conforme en aura besoin
    # pas la validation, elle a déjà servi à choisir la meilleure époque
    P_cal = predict(net, cal.X, mean, std)
    np.savez_compressed(out / "calibration.npz", P=P_cal, Y=cal.Y)
    best = max(history, key=lambda h: h["val_macro_auc"])
    print(f"best epoch {best['epoch']}  macro AUROC {best['val_macro_auc']:.4f}")

    # la seule fois où le test est utilisé
    P_test = predict(net, test.X, mean, std)
    res = evaluate(P_test, test.Y, P_cal, cal.Y, D.CLASSES)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    m = res["macro_auroc"]
    print(f"test macro AUROC {m['value']:.4f}  (95% CI {m['ci95'][0]:.4f} to {m['ci95'][1]:.4f})")


def cmd_export(a):
    run = Path(a.run)
    # weights_only=False car le fichier contient aussi mean et std (des tableaux numpy)
    # c'est notre propre fichier, on peut lui faire confiance
    ckpt = torch.load(run / "model.pt", weights_only=False)
    net = ECGResNet()
    net.load_state_dict(ckpt["state_dict"])
    cal = np.load(run / "calibration.npz")
    results = json.loads((run / "results.json").read_text())
    # les ECG d'exemple de l'app viennent du test, ce sont des patients que le modèle n'a jamais vus
    test = D.load_cache(Path(a.cache)).split(D.TEST_FOLD)
    P_test = predict(net, test.X, ckpt["mean"], ckpt["std"])
    write_bundle(Path(a.out), net, ckpt["mean"], ckpt["std"], cal["P"], cal["Y"], P_test, test.Y, test, results)
    print(f"bundle written to {a.out}")


def main():
    p = argparse.ArgumentParser(prog="ecgc")
    sub = p.add_subparsers(required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--ptbxl", required=True)
    pp.add_argument("--cache", required=True)
    pp.set_defaults(func=cmd_prepare)
    pt = sub.add_parser("train")
    pt.add_argument("--cache", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch", type=int, default=128)
    pt.add_argument("--lr", type=float, default=3e-3)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--threads", type=int, default=0)
    pt.set_defaults(func=cmd_train)
    pe = sub.add_parser("export")
    pe.add_argument("--run", required=True)
    pe.add_argument("--cache", required=True)
    pe.add_argument("--out", required=True)
    pe.set_defaults(func=cmd_export)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
