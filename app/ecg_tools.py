"""Petits outils autour du signal. Lire un ECG envoyé par l'utilisateur et mesurer le rythme."""

import io

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100
N = 1000


def read_csv_ecg(raw_bytes, fs):
    """Lit un CSV 12 dérivations et renvoie (signal 12 x 1000 en mV à 100 Hz, liste de remarques).

    Accepte 12 colonnes (une par dérivation) ou 12 lignes, avec ou sans en-tête.
    Lève ValueError avec un message clair si le fichier ne convient pas.
    """
    notes = []
    df = pd.read_csv(io.BytesIO(raw_bytes), header=None, sep=None, engine="python")

    # si la première ligne contient des noms de dérivations, on s'en sert pour remettre dans l'ordre
    first = [str(v).strip().upper() for v in df.iloc[0].tolist()]
    if set(l.upper() for l in LEADS) <= set(first):
        df = df.iloc[1:]
        df.columns = first
        df = df[[l.upper() for l in LEADS]]
    x = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    if x.shape[1] != 12 and x.shape[0] == 12:
        x = x.T  # le fichier était en lignes
    if x.shape[1] != 12:
        raise ValueError(f"Expected 12 leads, found a table of {x.shape[0]} x {x.shape[1]}.")
    if np.isnan(x).mean() > 0.01:
        raise ValueError("The file contains non-numeric values.")
    x = np.nan_to_num(x).T  # (12, temps)

    # remise à 100 Hz
    if fs != FS:
        x = resample_poly(x, FS, fs, axis=1)
        notes.append(f"Resampled from {fs} Hz to {FS} Hz.")

    if x.shape[1] < N:
        raise ValueError(f"The recording is {x.shape[1] / FS:.1f} s long, at least 10 s are needed.")
    if x.shape[1] > N:
        notes.append("Only the first 10 seconds are used.")
        x = x[:, :N]

    # beaucoup d'appareils exportent en microvolts
    if np.abs(x).max() > 50:
        x = x / 1000.0
        notes.append("Values looked like microvolts and were converted to millivolts.")
    return x.astype(np.float32), notes


def to_csv(signal):
    # même format que celui accepté à l'import, pratique comme exemple
    df = pd.DataFrame(np.asarray(signal).T, columns=LEADS)
    return df.to_csv(index=False, float_format="%.3f").encode()


def rhythm(signal, fs=FS):
    """Mesures grossières sur la dérivation II. Fréquence cardiaque, intervalle RR, régularité.

    Détection simple des QRS (dérivée au carré lissée puis pics), suffisante pour un ordre de grandeur.
    """
    x = np.asarray(signal[1], dtype=np.float64)
    energy = np.convolve(np.diff(x) ** 2, np.ones(int(0.12 * fs)) / (0.12 * fs), mode="same")
    if energy.max() <= 0:
        return None
    peaks, _ = find_peaks(energy, height=0.3 * energy.max(), distance=int(0.3 * fs))
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / fs
    cv = rr.std() / rr.mean()  # variation relative des intervalles RR
    return {
        "hr": 60.0 / np.median(rr),
        "rr_ms": 1000 * np.median(rr),
        "cv": cv,
        "regular": cv < 0.12,
        "beats": len(peaks),
    }
