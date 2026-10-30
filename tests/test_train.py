import numpy as np
import torch
from torch import nn

from ecgc import data as D
from ecgc.train import augment, fit, macro_auc, predict


def test_macro_auc_perfect_ranking():
    Y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    P = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.3]])
    assert macro_auc(Y, P) == 1.0


def test_augment_keeps_shape():
    x = torch.randn(4, 12, 1000)
    assert augment(x, torch.Generator().manual_seed(0)).shape == x.shape


class ConstantModel(nn.Module):
    # faux modèle qui donne un score très élevé aux 5 diagnostics, quel que soit l'ECG
    def forward(self, x):
        return torch.full((len(x), 5), 5.0)


def test_predict_gives_independent_probabilities():
    # un ECG peut avoir plusieurs diagnostics à la fois
    # si le modèle est sûr des 5, les 5 probabilités doivent être hautes
    P = predict(ConstantModel(), np.zeros((3, 12, 1000), dtype=np.float32), np.zeros(12), np.ones(12))
    assert (P > 0.9).all()


def test_fit_runs_on_fake_data(fake_cache):
    ds = D.load_cache(fake_cache)
    train, val = ds.split(D.TRAIN_FOLDS), ds.split(D.VAL_FOLD)
    mean, std = D.lead_stats(train.X)
    model, history = fit(train, val, mean, std, epochs=1, batch=32, log=lambda *_: None)
    assert len(history) == 1 and np.isfinite(history[0]["train_loss"])
