"""Lance le pipeline.

    python -m ecgc.run prepare --ptbxl chemin/vers/ptb-xl --cache data/ptbxl100.npz
    python -m ecgc.run train   --cache data/ptbxl100.npz --out runs/main
"""

import argparse
import json
from pathlib import Path

import torch

from . import data as D
from .train import fit


def cmd_prepare(a):
    print(f"Reading PTB-XL from {a.ptbxl}")
    D.build_cache(Path(a.ptbxl), Path(a.cache))
    print(f"Cache written to {a.cache}")


def cmd_train(a):
    torch.set_num_threads(a.threads or torch.get_num_threads())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = D.load_cache(Path(a.cache))
    train, test = ds.split(D.TRAIN_FOLDS), ds.split(D.TEST_FOLD)
    print(f"train {len(train)}  test {len(test)}")

    mean, std = D.lead_stats(ds.X)
    net, history = fit(train, test, mean, std, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed)

    # on sauve les poids et la normalisation ensemble, l'un ne sert à rien sans l'autre
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    best = max(history, key=lambda h: h["val_macro_auc"])
    print(f"best epoch {best['epoch']}  macro AUROC {best['val_macro_auc']:.4f}")


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
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
