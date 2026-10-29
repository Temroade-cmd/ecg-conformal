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
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # sortie = transformation + entrée d'origine
        return self.act(self.body(x) + x)
