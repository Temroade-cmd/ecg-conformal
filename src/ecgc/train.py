"""Entraînement du modèle."""

import math
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .model import ECGResNet


def macro_auc(Y, P):
    # moyenne des AUROC des 5 classes (on saute une classe si elle n'a qu'une seule valeur)
    return float(np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in range(Y.shape[1]) if 0 < Y[:, k].sum() < len(Y)]))


def augment(x, g):
    # petites perturbations réalistes sur l'ECG
    # amplitude un peu changée, dérive de la ligne de base, bruit, petit décalage dans le temps
    B, C, T = x.shape
    scale = 1.0 + 0.1 * (2 * torch.rand(B, 1, 1, generator=g) - 1)
    t = torch.linspace(0, 1, T).view(1, 1, T)
    freq = 0.1 + 0.4 * torch.rand(B, 1, 1, generator=g)
    phase = 2 * math.pi * torch.rand(B, 1, 1, generator=g)
    wander = 0.1 * torch.rand(B, 1, 1, generator=g) * torch.sin(2 * math.pi * freq * 10 * t + phase)
    noise = 0.02 * torch.randn(x.shape, generator=g)
    x = x * scale + wander + noise
    shift = int(torch.randint(-50, 51, (1,), generator=g))  # jusqu'à 0.5 s
    return torch.roll(x, shifts=shift, dims=-1)


@torch.no_grad()
def predict(model, X, mean, std, batch=256):
    # probas du modèle sur un tableau d'ECG, par paquets pour ne pas saturer la mémoire
    model.eval()
    m = torch.as_tensor(mean).view(1, -1, 1)
    s = torch.as_tensor(std).view(1, -1, 1)
    out = []
    for i in range(0, len(X), batch):
        xb = (torch.as_tensor(X[i : i + batch]) - m) / s
        out.append(torch.sigmoid(model(xb)).numpy())  # une proba indépendante par diagnostic
    return np.concatenate(out)


def fit(train, val, mean, std, epochs=30, batch=128, lr=3e-3, seed=0, log=print):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = ECGResNet(n_classes=train.Y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * math.ceil(len(train) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    loss_fn = nn.BCEWithLogitsLoss()  # multi-label, une sigmoïde et une perte binaire par diagnostic

    m = torch.as_tensor(mean).view(1, -1, 1)
    s = torch.as_tensor(std).view(1, -1, 1)
    Xtr, Ytr = torch.as_tensor(train.X), torch.as_tensor(train.Y)

    best_auc, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb = (augment(Xtr[idx], g) - m) / s
            loss = loss_fn(model(xb), Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)

        val_auc = macro_auc(val.Y, predict(model, val.X, mean, std))
        history.append({"epoch": ep, "train_loss": total / len(Xtr), "val_macro_auc": val_auc})
        log(f"epoch {ep:02d}  loss {total / len(Xtr):.4f}  val macro AUROC {val_auc:.4f}  ({time.time() - t0:.0f}s)")
        if val_auc > best_auc:  # on garde une copie des poids de la meilleure époque
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history
