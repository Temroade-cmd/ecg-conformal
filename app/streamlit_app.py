"""App de démo ECG Conformal.

    streamlit run app/streamlit_app.py

L'app télécharge le modèle publié en release sur GitHub (le modèle ONNX, les scores de calibration
et quelques ECG du test), puis utilise le même code conformal que l'évaluation (src/ecgc/conformal.py).
"""

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st

# Streamlit Cloud installe seulement app/requirements.txt, pas le package ecgc (il demanderait PyTorch)
# on ajoute donc src/ au chemin pour importer conformal.py directement, il n'a besoin que de numpy
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from ecgc import conformal as cp
import ecg_tools
import plots

REPO = "https://github.com/Temroade-cmd/ecg-conformal"
MODEL_TAG = os.environ.get("ECGC_MODEL_TAG", "model-v1")
MODEL_DIR = Path(os.environ.get("ECGC_MODEL_DIR", HERE / "model"))
FILES = ["ecg_resnet.onnx", "meta.json", "samples.json"]

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


def decide(probs, cal, alpha):
    # réponse conforme pour un ECG, et seuils pour dessiner les zones des barres
    inc_pos, inc_neg = cp.mondrian_sets(probs[None, :], cal, alpha)
    states = cp.states(inc_pos, inc_neg)[0]
    bounds = [cp.decision_bounds(c["pos"], c["neg"], alpha) for c in cal]
    return states, bounds


def figure(svg, alt=""):
    # le SVG est affiché comme une image, il garde ses proportions sur tous les écrans
    data = base64.b64encode(svg.encode()).decode()
    st.html(f'<img alt="{alt}" style="width:100%" src="data:image/svg+xml;base64,{data}">')


meta, samples, session, cal, X = load_bundle()
classes = meta["classes"]
names = [meta["class_names"][k] for k in classes]

st.title("ECG Conformal")
st.write("A 12-lead ECG classifier that says when it cannot tell.")

# un ECG envoyé par l'utilisateur, gardé dans la session
with st.expander("Use your own ECG"):
    st.markdown(
        "A CSV with 12 columns in the order I, II, III, aVR, aVL, aVF, V1 to V6 (or a header with those names), "
        "in millivolts, at least 10 seconds. Nothing is stored. "
        "Keep in mind the guarantee was calibrated on PTB-XL, another device or hospital may not respect it."
    )
    up_col, fs_col = st.columns([3, 1])
    with up_col:
        upload = st.file_uploader("ECG file", type=["csv", "txt"], label_visibility="collapsed")
    with fs_col:
        fs_in = st.selectbox("Sampling rate", [100, 250, 500, 1000], index=2, format_func=lambda v: f"{v} Hz")
    if upload is not None:
        try:
            sig, notes = ecg_tools.read_csv_ecg(upload.getvalue(), fs_in)
            st.session_state["mine"] = sig
            for n in notes:
                st.caption(n)
            st.caption("Loaded. Pick “your ECG” below.")
        except ValueError as e:
            st.error(str(e))

ids = [s["ecg_id"] for s in samples]
options = ids + (["your ECG"] if "mine" in st.session_state else [])
picked = st.selectbox("Record from the PTB-XL test fold", options)
if picked == "your ECG":
    signal, truth, title = st.session_state["mine"], None, "Your ECG"
else:
    i = ids.index(picked)
    signal = X[i]
    truth = [k in samples[i]["labels"] for k in classes]
    title = f"PTB-XL record {picked}"
probs = predict(session, signal)

c1, c2 = st.columns(2)
with c1:
    level = st.slider("Guarantee", min_value=80, max_value=99, value=90, format="%d%%")
with c2:
    target = st.selectbox("Diagnosis to inspect", range(len(classes)), index=int(np.argmax(probs)),
                          format_func=lambda k: names[k])
alpha = 1 - level / 100
states, bounds = decide(probs, cal, alpha)

# figure 1, l'ECG comme sur le papier
figure(plots.ecg_sheet(signal, meta["leads"], meta["fs"]), "12-lead ECG")
st.caption(f"Fig. 1. {title}. 10 s at 100 Hz, 25 mm/s, 10 mm/mV. The long strip at the bottom is lead II.")

# quelques mesures simples sur le rythme, et l'accord avec le cardiologue
m = ecg_tools.rhythm(signal)
decided = [k for k, s_ in enumerate(states) if s_ in ("present", "absent")]
agree = None if truth is None else sum((states[k] == "present") == truth[k] for k in decided)
cols = st.columns(4)
cols[0].metric("Heart rate, lead II", f"{m['hr']:.0f} bpm" if m else "n/a")
cols[1].metric("RR interval", f"{m['rr_ms']:.0f} ms" if m else "n/a", None if not m else ("regular" if m["regular"] else "irregular"),
               delta_color="off")
cols[2].metric(f"Answers given at {level}%", f"{len(decided)} of 5")
cols[3].metric("Agree with the cardiologist", f"{agree} of {len(decided)}" if agree is not None else "n/a")

# figure 2, la réponse par diagnostic, figure 3, où tombe cet ECG parmi la calibration
left, right = st.columns(2, gap="large")
with left:
    figure(plots.readout(probs, bounds, states, names, truth), "Model answer per diagnosis")
    st.caption(f"Fig. 2. Answer per diagnosis at {level}%. Grey and red zones are where the model commits, "
               "the hatched zone is where it cannot tell.")
with right:
    figure(plots.position(cal[target], probs[target], bounds[target], names[target]), "This ECG among calibration ECGs")
    st.caption(f"Fig. 3. {names[target]}. Scores of the calibration ECGs, with and without the diagnosis, "
               "and where this ECG falls.")
st.caption("Research demo, not a medical device. Data PTB-XL 1.0.3, PhysioNet, CC BY 4.0.")
