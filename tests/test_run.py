import json
import subprocess
import sys


def test_train_command(fake_cache, tmp_path):
    # la commande complète, comme elle sera lancée sur GitHub Actions
    subprocess.run([sys.executable, "-m", "ecgc.run", "train", "--cache", str(fake_cache), "--out", str(tmp_path),
                    "--epochs", "1", "--batch", "32"], check=True)
    assert (tmp_path / "model.pt").exists()
    assert len(json.loads((tmp_path / "history.json").read_text())) == 1
