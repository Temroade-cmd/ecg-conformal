"""App de démo ECG Conformal.

    streamlit run app/streamlit_app.py

L'app télécharge le modèle publié en release sur GitHub (le modèle ONNX, les scores de calibration
et quelques ECG du test), puis utilise le même code conformal que l'évaluation (src/ecgc/conformal.py).
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st

from ecgc import conformal as cp

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/Temroade-cmd/ecg-conformal"
MODEL_TAG = os.environ.get("ECGC_MODEL_TAG", "model-v1")
MODEL_DIR = Path(os.environ.get("ECGC_MODEL_DIR", HERE / "model"))
FILES = ["ecg_resnet.onnx", "meta.json", "samples.json"]
ANSWER = {"present": "present", "absent": "absent", "uncertain": "can't tell", "atypical": "atypical"}

st.set_page_config(page_title="ECG Conformal", layout="wide")


def download_model():
    # la première fois, on télécharge les fichiers de la release, ensuite ils restent sur le disque
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = MODEL_DIR / name
        if not target.exists():
            part = target.with_suffix(target.suffix + ".part")  # un téléchargement coupé ne laisse pas de fichier cassé
            urllib.request.urlretrieve(f"{REPO}/releases/download/{MODEL_TAG}/{name}", part)
            part.rename(target)


@st.cache_resource  # chargé une seule fois pour tous les visiteurs
def load_bundle():
    download_model()
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    samples = json.loads((MODEL_DIR / "samples.json").read_text())
    session = ort.InferenceSession(str(MODEL_DIR / "ecg_resnet.onnx"), providers=["CPUExecutionProvider"])
    cal = [{"pos": np.asarray(meta["calibration"][k]["pos"]), "neg": np.asarray(meta["calibration"][k]["neg"])}
           for k in meta["classes"]]
    X = np.asarray([s["signal"] for s in samples], dtype=np.float32)
    return meta, samples, session, cal, X


def predict(session, x):
    # l'ONNX contient la normalisation et la sigmoïde, il reçoit des millivolts et rend des probas
    return session.run(["prob"], {"ecg": x[None].astype(np.float32)})[0][0]


meta, samples, session, cal, X = load_bundle()
classes = meta["classes"]
names = [meta["class_names"][k] for k in classes]

st.title("ECG Conformal")
st.write("A 12-lead ECG classifier that says when it cannot tell.")

ids = [s["ecg_id"] for s in samples]
picked = st.selectbox("Record from the PTB-XL test fold", ids)
level = st.slider("Guarantee", min_value=80, max_value=99, value=90, format="%d%%")
alpha = 1 - level / 100

i = ids.index(picked)
probs = predict(session, X[i])
inc_pos, inc_neg = cp.mondrian_sets(probs[None], cal, alpha)
states = cp.states(inc_pos, inc_neg)[0]
truth = samples[i]["labels"]

st.table([
    {"Diagnosis": names[k], "Probability": f"{probs[k]:.2f}", "Answer": ANSWER[states[k]],
     "Cardiologist": "yes" if classes[k] in truth else "no"}
    for k in range(len(classes))
])
st.caption("Research demo, not a medical device. Data PTB-XL 1.0.3, PhysioNet, CC BY 4.0.")
