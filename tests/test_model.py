import torch

from ecgc.model import ECGResNet, ResBlock1d


def test_block_keeps_length():
    # le padding k // 2 garde la même longueur de signal
    block = ResBlock1d(64, 64)
    assert block(torch.randn(2, 64, 1000)).shape == (2, 64, 1000)


def test_model_output_shape():
    # 4 ECG de 12 dérivations sur 10 s, on attend 5 scores par ECG
    model = ECGResNet()
    assert model(torch.randn(4, 12, 1000)).shape == (4, 5)


def test_gradients_reach_first_layer():
    # vérifie que l'apprentissage peut remonter jusqu'à la toute première convolution
    model = ECGResNet()
    model(torch.randn(2, 12, 1000)).sum().backward()
    first = model.features[0].body[0].weight
    assert first.grad is not None and first.grad.abs().sum() > 0


def test_deployable_matches_manual_steps():
    from ecgc.model import Deployable

    net = ECGResNet().eval()
    mean, std = torch.rand(12), torch.rand(12) + 0.5
    x = torch.randn(3, 12, 1000)
    with torch.no_grad():
        expected = torch.sigmoid(net((x - mean.view(1, -1, 1)) / std.view(1, -1, 1)))
        got = Deployable(net, mean, std).eval()(x)
    assert torch.allclose(got, expected, atol=1e-6)
    assert ((got >= 0) & (got <= 1)).all()  # ce sont bien des probabilités
