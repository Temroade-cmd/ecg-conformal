"""Chargement de PTB-XL, labels des 5 superclasses et découpage en folds.

PTB-XL est déjà découpé en 10 folds (les folds 9 et 10 ont des labels vérifiés par des cardiologues).
    folds 1 à 7  entraînement
    fold 8       validation, pour choisir la meilleure époque
    fold 9       calibration conforme
    fold 10      test, utilisé une seule fois à la fin
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100  # fréquence en Hz (version basse résolution de PTB-XL)
N_SAMPLES = 1000  # 10 secondes

TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7)
VAL_FOLD = 8
CAL_FOLD = 9
TEST_FOLD = 10


@dataclass
class Dataset:
    X: np.ndarray  # signaux (N, 12, 1000) en millivolts
    Y: np.ndarray  # labels (N, 5), 1 si le diagnostic est présent
    folds: np.ndarray  # numéro de fold de chaque ECG
    ecg_ids: np.ndarray  # identifiant PTB-XL

    def split(self, folds):
        m = np.isin(self.folds, np.atleast_1d(folds))
        return Dataset(self.X[m], self.Y[m], self.folds[m], self.ecg_ids[m])

    def __len__(self):
        return len(self.ecg_ids)


def superdiagnostic_labels(meta, statements):
    # chaque code SCP (ex IMI, LVH...) appartient à une des 5 superclasses
    # comme dans le benchmark de Strodthoff et al. on prend tous les codes listés, peu importe leur score
    diag = statements[statements["diagnostic"] == 1]
    code_to_class = diag["diagnostic_class"].to_dict()
    Y = np.zeros((len(meta), len(CLASSES)), dtype=np.float32)
    for i, codes in enumerate(meta["scp_codes"]):
        if isinstance(codes, str):
            codes = ast.literal_eval(codes)  # la colonne est stockée comme un dict en texte
        for code in codes:
            cls = code_to_class.get(code)
            if cls in CLASSES:
                Y[i, CLASSES.index(cls)] = 1.0
    return Y


def read_metadata(root):
    root = Path(root)
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return meta, statements


def build_cache(root, out):
    # lire les 21k fichiers wfdb prend du temps, donc on le fait une fois et on sauve tout dans un .npz
    import wfdb

    root, out = Path(root), Path(out)
    meta, statements = read_metadata(root)
    Y = superdiagnostic_labels(meta, statements)
    keep = Y.sum(1) > 0  # on enlève les ECG sans aucun label (comme le benchmark)
    meta, Y = meta[keep], Y[keep]

    X = np.empty((len(meta), 12, N_SAMPLES), dtype=np.float16)  # float16 pour diviser la taille par 2
    for i, fname in enumerate(meta["filename_lr"]):
        sig, fields = wfdb.rdsamp(str(root / fname))
        # dans les fichiers PTB-XL les dérivations sont écrites AVR, AVL, AVF en majuscules
        names = [s.upper() for s in fields["sig_name"]]
        if names != [l.upper() for l in LEADS]:
            sig = sig[:, [names.index(l.upper()) for l in LEADS]]
        X[i] = np.nan_to_num(sig[:N_SAMPLES].T).astype(np.float16)
        if (i + 1) % 2000 == 0:
            print(f"  read {i + 1}/{len(meta)} records", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=X,
        Y=Y,
        folds=meta["strat_fold"].to_numpy(dtype=np.int16),
        ecg_ids=meta.index.to_numpy(dtype=np.int32),
    )
    return out


def load_cache(path):
    d = np.load(path)
    return Dataset(
        X=d["X"].astype(np.float32),
        Y=d["Y"].astype(np.float32),
        folds=d["folds"].astype(int),
        ecg_ids=d["ecg_ids"].astype(int),
    )


def lead_stats(X):
    # moyenne et écart-type par dérivation, calculés sur le train seulement
    mean = X.mean(axis=(0, 2)).astype(np.float32)
    std = X.std(axis=(0, 2)).astype(np.float32) + 1e-6
    return mean, std
