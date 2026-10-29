"""ResNet 1D pour ECG 12 dérivations, proche du modèle resnet1d_wang du benchmark PTB-XL."""

import torch
from torch import nn


class ResBlock1d(nn.Module):
    # 3 convolutions (noyaux 7, 5, 3) + raccourci résiduel
    def __init__(self, c_in, c_out, kernels=(7, 5, 3)):
        super().__init__()
        layers = []
        c = c_in
        for i, k in enumerate(kernels):
            layers += [nn.Conv1d(c, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out)]
            if i < len(kernels) - 1:
                layers.append(nn.ReLU(inplace=True))
            c = c_out
        self.body = nn.Sequential(*layers)
        # si le nombre de canaux change (12 -> 64 par exemple), on ne peut pas additionner directement
        # le raccourci passe alors par une conv 1x1 qui remet l'entrée au bon nombre de canaux
        self.skip = (
            nn.Identity()
            if c_in == c_out
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # sortie = transformation + entrée d'origine (adaptée si besoin)
        return self.act(self.body(x) + self.skip(x))


class ECGResNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, widths=(64, 128, 128), dropout=0.3):
        super().__init__()
        blocks, c = [], n_leads
        for w in widths:
            blocks.append(ResBlock1d(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, n_classes))

    def forward(self, x):  # x de taille (batch, 12, temps), déjà normalisé
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)  # pooling moyen + max collés ensemble
        return self.head(h)  # logits, pas encore de sigmoïde


class Deployable(nn.Module):
    # version pour l'export, on met la normalisation et la sigmoïde dedans
    # comme ça l'app donne directement des millivolts et reçoit des probas entre 0 et 1

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        # register_buffer, ces valeurs sont sauvegardées avec le modèle mais ne sont pas entraînées
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1))

    def forward(self, ecg):
        return torch.sigmoid(self.net((ecg - self.mean) / self.std))
