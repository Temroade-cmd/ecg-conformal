"""Crée un petit faux dataset avec exactement la même structure de fichiers que PTB-XL.

Sert aux tests, pour faire tourner tout le pipeline sans télécharger 1.7 Go.
Les signaux sont inventés et n'ont aucun sens médical.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # spelled as in PTB-XL headers
CODES = {"NORM": "NORM", "IMI": "MI", "NDT": "STTC", "CLBBB": "CD", "LVH": "HYP"}


def beat_train(rng, label_codes, n=1000, fs=100):
    # battements simplifiés, chaque "diagnostic" change un peu la forme pour que le modèle ait quelque chose à apprendre
    t = np.arange(n) / fs
    hr = rng.uniform(55, 95)
    rr = 60 / hr
    sig = np.zeros((n, 12))
    gains = rng.uniform(0.5, 1.5, 12) * np.where(np.arange(12) == 3, -1, 1)
    width = 0.012 * (2.5 if "CLBBB" in label_codes else 1)
    for b in np.arange(rng.uniform(0, rr), 10, rr):
        qrs = np.exp(-((t - b) ** 2) / (2 * width**2))
        twave = np.exp(-((t - b - 0.28) ** 2) / (2 * 0.04**2))
        tsign = -0.6 if "NDT" in label_codes else 0.3
        amp = 2.2 if "LVH" in label_codes else 1.0
        sig += np.outer(qrs * amp, gains) + np.outer(twave * tsign, np.abs(gains))
        if "IMI" in label_codes:
            sig[:, [1, 2, 5]] += 0.25 * np.exp(-((t - b - 0.12) ** 2) / (2 * 0.05**2))[:, None]
    sig += 0.03 * rng.standard_normal(sig.shape)
    return sig.astype(np.float64)


def main(out: Path, n: int, seed: int):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ecg_id in range(1, n + 1):
        k = rng.choice(len(CODES), size=rng.choice([1, 1, 1, 2]), replace=False)
        codes = [list(CODES)[i] for i in k]
        if "NORM" in codes and len(codes) > 1:
            codes = [c for c in codes if c != "NORM"]
        rel = f"records100/{(ecg_id // 1000) * 1000:05d}/{ecg_id:05d}_lr"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        wfdb.wrsamp(
            record_name=Path(rel).name,
            fs=100,
            units=["mV"] * 12,
            sig_name=LEADS,
            p_signal=beat_train(rng, codes),
            fmt=["16"] * 12,
            write_dir=str((out / rel).parent),
        )
        rows.append(
            {
                "ecg_id": ecg_id,
                "scp_codes": str({c: 100.0 for c in codes} | {"SR": 0.0}),
                "strat_fold": (ecg_id - 1) % 10 + 1,
                "filename_lr": rel,
            }
        )
    pd.DataFrame(rows).set_index("ecg_id").to_csv(out / "ptbxl_database.csv")
    st = pd.DataFrame(
        {"diagnostic": [1.0] * len(CODES) + [np.nan], "diagnostic_class": list(CODES.values()) + [np.nan]},
        index=list(CODES) + ["SR"],
    )
    st.to_csv(out / "scp_statements.csv")
    print(f"fixture with {n} records in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/fixture"))
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.out, a.n, a.seed)
