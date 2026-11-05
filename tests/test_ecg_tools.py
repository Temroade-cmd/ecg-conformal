import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import ecg_tools as T


def _fake_ecg(hr=75, fs=100, seconds=10):
    # un QRS en forme de pic toutes les 60/hr secondes, sur les 12 dérivations
    t = np.arange(fs * seconds) / fs
    x = np.zeros_like(t)
    for b in np.arange(0.3, seconds, 60 / hr):
        x += np.exp(-((t - b) ** 2) / (2 * 0.015**2))
    return np.tile(x, (12, 1)).astype(np.float32)


def test_heart_rate():
    r = T.rhythm(_fake_ecg(hr=75))
    assert abs(r["hr"] - 75) < 2 and r["regular"]


def test_csv_round_trip():
    x = _fake_ecg()
    y, notes = T.read_csv_ecg(T.to_csv(x), 100)
    assert np.allclose(x, y, atol=1e-3) and notes == []


def test_500hz_microvolts_in_rows():
    # 12 lignes, pas d'en-tête, 500 Hz, en microvolts
    x = np.repeat(_fake_ecg(), 5, axis=1) * 1000
    buf = io.StringIO()
    np.savetxt(buf, x, delimiter=",")
    y, notes = T.read_csv_ecg(buf.getvalue().encode(), 500)
    assert y.shape == (12, 1000) and np.abs(y).max() < 5 and len(notes) == 2


def test_too_short():
    with pytest.raises(ValueError):
        T.read_csv_ecg(T.to_csv(_fake_ecg(seconds=5)), 100)
