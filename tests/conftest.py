import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fake_cache(tmp_path_factory):
    # un faux PTB-XL de 300 ECG, lu une seule fois pour tous les tests qui en ont besoin
    from ecgc import data as D

    tmp = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_fixture.py"), "--out", str(tmp / "ptbxl"), "--n", "300"], check=True)
    D.build_cache(tmp / "ptbxl", tmp / "cache.npz")
    return tmp / "cache.npz"
