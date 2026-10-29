"""Entraînement du modèle."""

import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
